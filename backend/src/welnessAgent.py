import logging
import json
from pathlib import Path
from typing import Annotated
from datetime import datetime

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
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class WellnessCompanion(Agent):
    def __init__(self, previous_entries: list = None) -> None:
        # Load previous check-in history
        self.previous_entries = previous_entries or []
        
        # Build context from previous entries for the system prompt
        history_context = self._build_history_context()
        
        super().__init__(
            instructions=f"""You are a supportive health and wellness companion who conducts daily check-ins with users. Your role is to be warm, empathetic, and grounded - not a therapist or medical professional.

Your approach:
- Have natural, conversational check-ins that feel like talking to a caring friend
- Ask open-ended questions about mood, energy, and daily intentions
- Listen actively and reflect back what you hear
- Offer small, practical, actionable suggestions when appropriate
- Never diagnose, prescribe, or give medical advice
- Keep the conversation brief and focused (5-10 minutes)

During each check-in, naturally gather:
1. How they're feeling today (mood and energy level)
2. What's on their mind or causing stress (if anything)
3. Their main objectives or intentions for the day (1-3 things)
4. One small self-care action they might take

Guidelines for advice:
- Keep suggestions small and realistic: "Maybe take a 5-minute walk" not "Exercise for an hour"
- Break big goals into smaller steps
- Encourage simple grounding practices (breathing, short breaks, movement)
- Validate their feelings without judgment
- Never claim to diagnose conditions or prescribe treatments

Conversation flow:
1. Start with a warm greeting
2. Ask about their current mood and energy
3. Explore what's on their mind today
4. Discuss their intentions/goals for the day
5. Offer a small piece of grounded advice or reflection
6. Recap what you heard and confirm accuracy
7. Once confirmed, use the save_checkin tool to save the session

{history_context}

Remember: Keep responses conversational and natural for voice interaction. Avoid bullet points or complex formatting."""
        )
        
        # Initialize current session state
        self.current_session = {
            "mood": None,
            "energy": None,
            "stress_factors": [],
            "objectives": [],
            "self_care_action": None,
            "agent_summary": None
        }

    def _build_history_context(self) -> str:
        """Build context string from previous check-ins."""
        if not self.previous_entries:
            return "This is your first check-in with this user."
        
        # Get the most recent entry
        last_entry = self.previous_entries[-1]
        days_ago = self._calculate_days_ago(last_entry.get("timestamp"))
        
        context = f"\nPrevious check-in history:\n"
        context += f"Last check-in was {days_ago}. "
        context += f"They reported feeling: {last_entry.get('mood', 'not recorded')}. "
        
        if last_entry.get('objectives'):
            context += f"Their goals were: {', '.join(last_entry['objectives'])}. "
        
        # Include a few more recent entries if available
        if len(self.previous_entries) > 1:
            context += f"\nTotal check-ins completed: {len(self.previous_entries)}. "
            
            # Look for patterns in recent mood
            recent_moods = [e.get('mood', '') for e in self.previous_entries[-3:] if e.get('mood')]
            if recent_moods:
                context += f"Recent mood trend: {', '.join(recent_moods)}. "
        
        context += "\nReference this history naturally in your conversation to show continuity and care."
        
        return context
    
    def _calculate_days_ago(self, timestamp_str: str) -> str:
        """Calculate how long ago a timestamp was."""
        try:
            last_time = datetime.fromisoformat(timestamp_str)
            now = datetime.now()
            days = (now - last_time).days
            
            if days == 0:
                return "earlier today"
            elif days == 1:
                return "yesterday"
            else:
                return f"{days} days ago"
        except:
            return "recently"

    @function_tool
    async def save_checkin(
        self,
        context: RunContext,
        mood: Annotated[str, "User's self-reported mood (e.g., 'tired but optimistic', 'stressed', 'energized')"],
        energy_level: Annotated[str, "User's energy level (e.g., 'low', 'medium', 'high', or descriptive)"],
        stress_factors: Annotated[list[str], "List of things causing stress or on their mind (can be empty)"],
        objectives: Annotated[list[str], "List of 1-3 main goals or intentions for the day"],
        self_care_action: Annotated[str, "One small self-care or wellness action they plan to take"],
        agent_summary: Annotated[str, "Brief 1-2 sentence summary of the check-in and your reflection"]
    ):
        """Save the completed wellness check-in to the JSON log. Use this only after confirming the recap with the user.
        
        Args:
            mood: User's current mood in their own words
            energy_level: How energized or drained they feel
            stress_factors: Things causing stress or concern (empty list if none)
            objectives: Their main goals or intentions for today (1-3 items)
            self_care_action: One small wellness action they'll take
            agent_summary: Your brief summary of the session and reflection
        """
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "mood": mood,
            "energy_level": energy_level,
            "stress_factors": stress_factors if stress_factors else [],
            "objectives": objectives,
            "self_care_action": self_care_action,
            "agent_summary": agent_summary
        }
        
        # Update internal state
        self.current_session = entry
        
        # Load existing log
        log_file = Path("wellness_log.json")
        
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {"check_ins": []}
        
        # Append new entry
        log_data["check_ins"].append(entry)
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Check-in saved: {entry}")
        
        # Return confirmation message
        objectives_str = ", ".join(objectives[:3])  # Limit to first 3
        stress_str = f" You mentioned feeling stressed about {', '.join(stress_factors[:2])}." if stress_factors else ""
        
        return f"Thank you for checking in today! I've recorded that you're feeling {mood} with {energy_level} energy.{stress_str} Your main focus is: {objectives_str}. And you're planning to {self_care_action}. I'm here whenever you need to talk. Take care!"


def load_wellness_history() -> list:
    """Load previous check-in history from JSON file."""
    log_file = Path("wellness_log.json")
    
    if not log_file.exists():
        return []
    
    try:
        with open(log_file, 'r') as f:
            log_data = json.load(f)
            return log_data.get("check_ins", [])
    except Exception as e:
        logger.error(f"Error loading wellness history: {e}")
        return []


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Load previous check-in history
    previous_entries = load_wellness_history()
    logger.info(f"Loaded {len(previous_entries)} previous check-ins")

    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(
            model="gemini-2.5-flash",
        ),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with our wellness companion agent
    await session.start(
        agent=WellnessCompanion(previous_entries=previous_entries),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))