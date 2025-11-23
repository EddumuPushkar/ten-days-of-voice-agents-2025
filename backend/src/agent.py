import logging
import json
from pathlib import Path
from typing import Annotated

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


class BaristaAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly barista at Sunrise Coffee Co, a cozy neighborhood coffee shop known for exceptional drinks and warm service.
            
            Your job is to take the customer's coffee order via voice conversation. Be warm, enthusiastic, and helpful!
            
            You need to collect the following information for each order:
            1. Drink type (e.g., latte, cappuccino, americano, cold brew, mocha, etc.)
            2. Size (small, medium, or large)
            3. Milk preference (whole, skim, oat, almond, soy, or none for black coffee)
            4. Any extras (e.g., extra shot, vanilla syrup, caramel drizzle, whipped cream, etc.)
            5. Customer's name for the order
            
            Guidelines:
            - Start by greeting the customer warmly and asking what they'd like to order
            - Ask clarifying questions one at a time to make the conversation natural
            - If the customer mentions multiple items at once, acknowledge them and ask about any missing details
            - Confirm the order before finalizing
            - Once you have all the information, use the save_order tool to save it
            - Keep responses conversational and friendly, without complex formatting
            - If a customer asks for recommendations, suggest popular items
            
            Remember: You're speaking to customers via voice, so keep it natural and conversational!""",
        )
        
        # Initialize order state
        self.order_state = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None
        }

    @function_tool
    async def save_order(
        self,
        context: RunContext,
        drink_type: Annotated[str, "The type of coffee drink ordered"],
        size: Annotated[str, "The size of the drink (small, medium, or large)"],
        milk: Annotated[str, "The type of milk or 'none' for black coffee"],
        extras: Annotated[list[str], "List of any extras or modifications"],
        name: Annotated[str, "Customer's name for the order"]
    ):
        """Save the completed coffee order to a JSON file. Use this tool only when you have collected all order details from the customer.
        
        Args:
            drink_type: The type of coffee drink (e.g., latte, cappuccino, americano)
            size: The size of the drink (small, medium, or large)
            milk: The milk preference (whole, skim, oat, almond, soy, or none)
            extras: List of extras like extra shot, syrups, whipped cream, etc.
            name: The customer's name for the order
        """
        
        order = {
            "drinkType": drink_type,
            "size": size,
            "milk": milk,
            "extras": extras if extras else [],
            "name": name
        }
        
        # Update internal state
        self.order_state = order
        
        # Create orders directory if it doesn't exist
        orders_dir = Path("orders")
        orders_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = orders_dir / f"order_{name.lower().replace(' ', '_')}_{timestamp}.json"
        
        # Save order to file
        with open(filename, 'w') as f:
            json.dump(order, f, indent=2)
        
        logger.info(f"Order saved: {order}")
        
        # Return confirmation message
        extras_str = f" with {', '.join(extras)}" if extras else ""
        milk_str = f" with {milk} milk" if milk.lower() != "none" else ""
        
        return f"Perfect! I've got your order saved: {size} {drink_type}{milk_str}{extras_str} for {name}. Your order will be ready in just a few minutes!"


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

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

    # Start the session with our barista agent
    await session.start(
        agent=BaristaAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))