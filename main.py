"""VoxClaude — voice AI assistant for Discord.

Pipeline: voice receive -> faster-whisper (local) -> Gemini API -> edge-tts -> voice reply.
Wake word: "Клод". Commands: /join, /leave, /ask, /learn.
"""

import asyncio
import glob
import logging
import os
import re
import tempfile
import time
from collections import deque

from dotenv import load_dotenv

load_dotenv()

_here = os.path.dirname(os.path.abspath(__file__))
for _d in glob.glob(os.path.join(_here, ".venv", "Lib", "site-packages", "nvidia", "*", "bin")):
    try:
        os.add_dll_directory(_d)
    except OSError:
        pass
    os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import discord
from discord import app_commands
from discord.ext import commands, voice_recv
from discord.ext.voice_recv.opus import PacketDecoder
from discord.opus import OpusError
from faster_whisper import WhisperModel
from google import genai
from google.genai import types as genai_types
import edge_tts

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voxclaude")
logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
TTS_VOICE = os.environ.get("TTS_VOICE", "ru-RU-DmitryNeural")
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")

SILENCE_FLUSH_SEC = 0.8
MIN_PHRASE_SEC = 0.6
MAX_PHRASE_SEC = 25
KNOWLEDGE_FILE = os.environ.get("KNOWLEDGE_FILE", "knowledge.txt")
KNOWLEDGE_LIMIT = 8000
HISTORY_TURNS = 6
GEMINI_TIMEOUT = 25
GEMINI_RETRIES = 3

SYSTEM_PROMPT = """Ты — Клод, голосовой игровой помощник в Discord-войсе. Отвечай на русском и английском, в зависимости от того, на каком языке с тобой разговаривают.
Отвечай неодносложно, но и не долго (от 3 до 6 предложений), по делу, разговорным языком с юмором — твой ответ будет озвучен вслух.
Не используй markdown, списки, эмодзи и спецсимволы — только обычные предложения.
Если вопрос про игру и в базе знаний ниже есть релевантные факты — опирайся на них в первую очередь.
Если не знаешь актуальную мету/патч — честно скажи и дай общий совет где можно посмотреть(обычно это телегам-каналы по играм, учти это).

=== БАЗА ЗНАНИЙ (пополняется игроками через /learn) ===
{knowledge}
"""

_orig_pop_data = PacketDecoder.pop_data


def _safe_pop_data(self, *, timeout: float = 0):
    try:
        return _orig_pop_data(self, timeout=timeout)
    except OpusError:
        log.warning("Corrupted opus packet (ssrc=%s), resetting decoder", getattr(self, "ssrc", "?"))
        try:
            self.reset()
        except Exception:
            pass
        return None


PacketDecoder.pop_data = _safe_pop_data

log.info("Loading Whisper (%s, device=%s)...", WHISPER_MODEL, WHISPER_DEVICE)
try:
    whisper = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type="auto")
except Exception as e:
    log.warning("GPU init failed (%s), falling back to CPU int8", e)
    whisper = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

gemini = genai.Client(api_key=GEMINI_API_KEY)


def load_knowledge() -> str:
    try:
        with open(KNOWLEDGE_FILE, encoding="utf-8") as f:
            data = f.read().strip()
        return data[-KNOWLEDGE_LIMIT:] if data else "(пока пусто)"
    except FileNotFoundError:
        return "(пока пусто)"


def add_knowledge(fact: str):
    with open(KNOWLEDGE_FILE, "a", encoding="utf-8") as f:
        f.write(fact.strip() + "\n")


class PhraseSink(voice_recv.AudioSink):
    def __init__(self, bot: "VoxClaude", guild_id: int):
        super().__init__()
        self.bot = bot
        self.guild_id = guild_id
        self.buffers: dict[int, bytearray] = {}
        self.last_packet: dict[int, float] = {}

    def wants_opus(self) -> bool:
        return False

    def write(self, user, data: voice_recv.VoiceData):
        if user is None or user.bot:
            return
        buf = self.buffers.setdefault(user.id, bytearray())
        if len(buf) < MAX_PHRASE_SEC * 192_000:
            buf.extend(data.pcm)
        self.last_packet[user.id] = time.monotonic()

    def cleanup(self):
        self.buffers.clear()
        self.last_packet.clear()

    def pop_finished(self) -> list[tuple[int, bytes]]:
        now = time.monotonic()
        done = []
        for uid, ts in list(self.last_packet.items()):
            if now - ts >= SILENCE_FLUSH_SEC and self.buffers.get(uid):
                done.append((uid, bytes(self.buffers.pop(uid))))
                self.last_packet.pop(uid, None)
        return done


def pcm_to_float16k(pcm: bytes) -> np.ndarray:
    audio = np.frombuffer(pcm, dtype=np.int16).reshape(-1, 2)
    mono = audio.mean(axis=1)
    return (mono[::3] / 32768.0).astype(np.float32)


SEARCH_TRIGGERS = re.compile(
    r"патч|нерф|баф|обнов|актуальн|последн|сейчас|недавн|вышел|вышл|релиз|новост",
    re.IGNORECASE,
)
GOOGLE_PREFIX = re.compile(r"^(за|по)?гугли(ть)?\b[\s,:!-]*", re.IGNORECASE)
WAKE_RE = re.compile(r"\b(клод|клоуд|клауд|клот|claude|cloud)\b[\s,.:!?-]*", re.IGNORECASE)


def detect_search(question: str) -> tuple[str, bool]:
    m = GOOGLE_PREFIX.match(question.strip())
    if m:
        return question.strip()[m.end():].strip(), True
    return question, bool(SEARCH_TRIGGERS.search(question))


def strip_wake_word(text: str) -> str | None:
    m = WAKE_RE.search(text)
    return text[m.end():].strip() if m else None


class VoxClaude(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.voice_states = True
        super().__init__(command_prefix="!", intents=intents)
        self.sinks: dict[int, PhraseSink] = {}
        self.history: dict[int, deque] = {}
        self.busy: dict[int, asyncio.Lock] = {}

    async def setup_hook(self):
        await self.tree.sync()
        self.loop.create_task(self.flush_loop())

    async def flush_loop(self):
        await self.wait_until_ready()
        while not self.is_closed():
            for gid, sink in list(self.sinks.items()):
                for uid, pcm in sink.pop_finished():
                    if len(pcm) >= MIN_PHRASE_SEC * 192_000:
                        asyncio.create_task(self.handle_phrase(gid, uid, pcm))
            await asyncio.sleep(0.2)

    async def handle_phrase(self, guild_id: int, user_id: int, pcm: bytes):
        audio = pcm_to_float16k(pcm)
        text = await asyncio.to_thread(self.transcribe, audio)
        if not text:
            return
        log.info("[%s] %s: %s", guild_id, user_id, text)

        question = strip_wake_word(text)
        if question is None or len(question) < 3:
            return

        lock = self.busy.setdefault(guild_id, asyncio.Lock())
        if lock.locked():
            log.info("[%s] skipping phrase, still answering previous one", guild_id)
            return
        async with lock:
            answer = await self.ask_gemini(guild_id, question)
            if answer:
                await self.speak(guild_id, answer)

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = whisper.transcribe(
            audio, language="ru", beam_size=1, vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def ask_gemini(self, guild_id: int, question: str) -> str:
        question, use_search = detect_search(question)
        if not question:
            return "А что загуглить-то?"
        hist = self.history.setdefault(guild_id, deque(maxlen=HISTORY_TURNS * 2))
        prompt_parts = [SYSTEM_PROMPT.format(knowledge=load_knowledge())]
        for role, msg in hist:
            prompt_parts.append(f"{role}: {msg}")
        prompt_parts.append(f"Игрок: {question}\nКлод:")

        config = None
        if use_search:
            log.info("[%s] using Google Search grounding: %s", guild_id, question)
            config = genai_types.GenerateContentConfig(
                tools=[genai_types.Tool(google_search=genai_types.GoogleSearch())]
            )

        answer = None
        for attempt in range(GEMINI_RETRIES):
            try:
                resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        gemini.models.generate_content,
                        model=GEMINI_MODEL,
                        contents="\n".join(prompt_parts),
                        config=config,
                    ),
                    timeout=GEMINI_TIMEOUT,
                )
                answer = (resp.text or "").strip()
                break
            except asyncio.TimeoutError:
                log.error("Gemini timeout (%ss)", GEMINI_TIMEOUT)
                return "Мозг не ответил вовремя, проверь VPN."
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    log.error("Gemini quota exceeded: %s", e)
                    return "Дневной лимит запросов исчерпан, приходи завтра."
                if "503" in msg or "UNAVAILABLE" in msg:
                    log.warning("Gemini overloaded (attempt %s/%s)", attempt + 1, GEMINI_RETRIES)
                    await asyncio.sleep(2)
                    continue
                log.error("Gemini error: %s", e)
                return "Не смог достучаться до мозга, попробуй ещё раз."
        if answer is None:
            return "Гугл перегружен, спроси через минутку."

        hist.append(("Игрок", question))
        hist.append(("Клод", answer))
        return answer

    async def speak(self, guild_id: int, text: str):
        guild = self.get_guild(guild_id)
        vc = guild.voice_client if guild else None
        if vc is None or not vc.is_connected():
            return
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            path = tmp.name
        try:
            await edge_tts.Communicate(text, TTS_VOICE).save(path)
            done = asyncio.Event()
            vc.play(
                discord.FFmpegPCMAudio(path, executable=FFMPEG_PATH),
                after=lambda err: self.loop.call_soon_threadsafe(done.set),
            )
            await done.wait()
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


bot = VoxClaude()


@bot.tree.command(name="join", description="Позвать Клода в твой войс (начнёт слушать wake-word)")
async def join(interaction: discord.Interaction):
    if not isinstance(interaction.user, discord.Member) or not interaction.user.voice:
        await interaction.response.send_message("Сначала зайди в войс.", ephemeral=True)
        return
    channel = interaction.user.voice.channel
    vc = interaction.guild.voice_client
    if vc is not None:
        await vc.move_to(channel)
    else:
        vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
    sink = PhraseSink(bot, interaction.guild_id)
    bot.sinks[interaction.guild_id] = sink
    vc.listen(sink)
    await interaction.response.send_message(
        f"🎙 Слушаю в {channel.mention}. Обращайся: **«Клод, …»**\n"
        f"Все в канале в курсе, что войс обрабатывается ботом? 🙂"
    )


@bot.tree.command(name="leave", description="Выгнать Клода из войса")
async def leave(interaction: discord.Interaction):
    vc = interaction.guild.voice_client
    if vc:
        bot.sinks.pop(interaction.guild_id, None)
        await vc.disconnect()
        await interaction.response.send_message("👋 Вышел.")
    else:
        await interaction.response.send_message("Я и так не в войсе.", ephemeral=True)


@bot.tree.command(name="learn", description="Научить Клода факту про игру")
@app_commands.describe(факт="Например: против Динамо в ган-билде бери Metal Skin")
async def learn(interaction: discord.Interaction, факт: str):
    add_knowledge(f"[{interaction.user.display_name}] {факт}")
    await interaction.response.send_message(f"🧠 Запомнил: «{факт}»")


@bot.tree.command(name="ask", description="Спросить Клода текстом (ответит голосом, если в войсе)")
async def ask(interaction: discord.Interaction, вопрос: str):
    await interaction.response.defer()
    answer = await bot.ask_gemini(interaction.guild_id, вопрос)
    await interaction.followup.send(answer)
    vc = interaction.guild.voice_client
    if vc and vc.is_connected() and not vc.is_playing():
        await bot.speak(interaction.guild_id, answer)


@bot.event
async def on_voice_state_update(member, before, after):
    if member.bot:
        return
    vc = member.guild.voice_client
    if vc and vc.channel and not [m for m in vc.channel.members if not m.bot]:
        bot.sinks.pop(member.guild.id, None)
        await vc.disconnect()


bot.run(DISCORD_TOKEN)
