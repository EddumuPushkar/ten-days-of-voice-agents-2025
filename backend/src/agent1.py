import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    # function_tool,
    # RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.agents import function_tool, RunContext
import json

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Load the tutor content
def load_tutor_content():
    try:
        with open("shared-data/day4_tutor_content.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("Tutor content file not found, using default content")
        return [
            {
                "id": "variables",
                "title": "Variables",
                "summary": "Variables store values so you can reuse them later.",
                "sample_question": "What is a variable and why is it useful?"
            },
            {
                "id": "loops",
                "title": "Loops",
                "summary": "Loops let you repeat an action multiple times.",
                "sample_question": "Explain the difference between a for loop and a while loop."
            }
        ]

TUTOR_CONTENT = load_tutor_content()


class GreeterAgent(Agent):
    """Greets user and routes to correct mode"""
    def __init__(self):
        concepts_list = ", ".join([c['title'] for c in TUTOR_CONTENT])
        super().__init__(
            instructions=f"""
You are a friendly learning assistant. 

Available concepts: {concepts_list}

Greet the user warmly and ask: "Which mode would you like? Say 'learn', 'quiz', or 'teach back'."

When they choose:
- If they say "learn" → call switch_to_learn()
- If they say "quiz" → call switch_to_quiz()  
- If they say "teach back" → call switch_to_teach_back()

Keep it simple and friendly!
            """,
        )

    @function_tool
    async def switch_to_learn(self, context: RunContext):
        """Switch to learn mode (Matthew's voice)"""
        logger.info("Switching to learn mode")
        return LearnAgent()  # Return the new agent directly

    @function_tool
    async def switch_to_quiz(self, context: RunContext):
        """Switch to quiz mode (Alicia's voice)"""
        logger.info("Switching to quiz mode")
        return QuizAgent()  # Return the new agent directly

    @function_tool
    async def switch_to_teach_back(self, context: RunContext):
        """Switch to teach back mode (Ken's voice)"""
        logger.info("Switching to teach back mode")
        return TeachBackAgent()  # Return the new agent directly


class LearnAgent(Agent):
    """Explains concepts - Matthew's voice"""
    def __init__(self):
        concepts_info = "\n".join([f"- {c['title']}: {c['summary']}" for c in TUTOR_CONTENT])
        super().__init__(
            instructions=f"""
You are Matthew, a patient teacher explaining programming concepts.

Available concepts:
{concepts_info}

When user asks about a concept, explain it clearly using the summary.

If they want to switch modes:
- "quiz me" → call switch_to_quiz()
- "let me teach you" → call switch_to_teach_back()

Keep explanations simple and encouraging!
            """,
            # Override TTS to use Matthew's voice
            tts=murf.TTS(
                voice="en-US-matthew",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        )

    # To add tools, use the @function_tool decorator.
    # Here's an example that adds a simple weather tool.
    # You also have to add `from livekit.agents import function_tool, RunContext` to the top of this file
    # @function_tool
    # async def lookup_weather(self, context: RunContext, location: str):
    #     """Use this tool to look up current weather information in the given location.
    #
    #     If the location is not supported by the weather service, the tool will indicate this. You must tell the user the location's weather is unavailable.
    #
    #     Args:
    #         location: The location to look up weather information for (e.g. city name)
    #     """
    #
    #     logger.info(f"Looking up weather for {location}")
    #
    #     return "sunny with a temperature of 70 degrees."

    @function_tool
    async def switch_to_quiz(self, context: RunContext):
        """Switch to quiz mode"""
        return QuizAgent()

    @function_tool
    async def switch_to_teach_back(self, context: RunContext):
        """Switch to teach back mode"""
        return TeachBackAgent()


class QuizAgent(Agent):
    """Quizzes the user - Alicia's voice"""
    def __init__(self):
        questions = "\n".join([f"- {c['title']}: {c['sample_question']}" for c in TUTOR_CONTENT])
        super().__init__(
            instructions=f"""
You are Alicia, an encouraging quiz master.

Available questions:
{questions}

Ask which concept they want to be quizzed on, then ask the question.
Listen to their answer and give brief positive feedback.

If they want to switch modes:
- "explain it to me" → call switch_to_learn()
- "let me teach you" → call switch_to_teach_back()

Be supportive and positive!
            """,
            # Override TTS to use Alicia's voice
            tts=murf.TTS(
                voice="en-US-alicia",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        )

    @function_tool
    async def switch_to_learn(self, context: RunContext):
        """Switch to learn mode"""
        return LearnAgent()

    @function_tool
    async def switch_to_teach_back(self, context: RunContext):
        """Switch to teach back mode"""
        return TeachBackAgent()


class TeachBackAgent(Agent):
    """Listens to user explanations - Ken's voice"""
    def __init__(self):
        concepts_list = ", ".join([c['title'] for c in TUTOR_CONTENT])
        super().__init__(
            instructions=f"""
You are Ken, an attentive listener who provides feedback.

Available concepts: {concepts_list}

Ask which concept they want to teach you.
Listen carefully to their explanation.
Give kind, constructive feedback on what they explained well.

If they want to switch modes:
- "explain it to me" → call switch_to_learn()
- "quiz me" → call switch_to_quiz()

Be encouraging!
            """,
            # Override TTS to use Ken's voice
            tts=murf.TTS(
                voice="en-US-ken",
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        )

    @function_tool
    async def switch_to_learn(self, context: RunContext):
        """Switch to learn mode"""
        return LearnAgent()

    @function_tool
    async def switch_to_quiz(self, context: RunContext):
        """Switch to quiz mode"""
        return QuizAgent()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using Google, Murf, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        # Note: Each agent can override this with their own voice
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session with the initial agent (GreeterAgent)
    # When a tool returns an Agent instance, the session automatically switches to that agent
    await session.start(
        agent=GreeterAgent(),  # Start with the greeter
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))