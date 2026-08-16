from __future__ import annotations

"""
AI Actor — Main Entry Point.

Starts the AI Actor system in either text mode or full audio mode.
"""

import argparse
import asyncio
import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel

from src.core.config import load_config
from src.core.logging_utils import DeferrableHandler, set_console_handler
from src.core.orchestrator import Orchestrator
from src.core.registry import list_providers


def setup_logging(level: str = "INFO") -> None:
    """Configure rich logging (deferrable, so logs don't cut into
    a response while it is streaming to the console)."""
    handler = DeferrableHandler(RichHandler(rich_tracebacks=True, markup=True))
    set_console_handler(handler)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[handler],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🎭 AI Actor — Theatrical AI Performance System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                           # Start in text mode
  python -m src.main --mode text               # Explicit text mode
  python -m src.main --config config/dev.yaml   # Custom config
  python -m src.main --list-providers           # Show available providers

Environment Variables:
  AI_ACTOR__LLM__MODEL=qwen2.5:32b    # Override LLM model
  AI_ACTOR__STT__MODEL=large-v3       # Override STT model
  AI_ACTOR_ENV=development            # Load development.yaml overrides
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["text", "audio"],
        default="text",
        help="Interaction mode (default: text)",
    )
    parser.add_argument(
        "--config",
        default="config",
        help="Config directory path (default: config/)",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="Environment name for config overrides",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Log level override",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start the performance from hour 0 (ignore saved performance state)",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="List available providers and exit",
    )

    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    """Main async entry point."""
    console = Console()

    # Load configuration
    config = load_config(config_dir=args.config, environment=args.env)
    if args.fresh:
        config.setdefault("system", {})["fresh_start"] = True

    # Setup logging
    log_level = args.log_level or config.get("system", {}).get("log_level", "INFO")
    setup_logging(log_level)

    # Show startup banner
    console.print(Panel.fit(
        "[bold magenta]🎭 AI Actor[/bold magenta]\n"
        f"[dim]Mode: {args.mode} | "
        f"LLM: {config.get('llm', {}).get('model', '?')} | "
        f"TTS: {config.get('tts', {}).get('provider', '?')}[/dim]",
        border_style="magenta",
    ))

    # Create and setup orchestrator
    orchestrator = Orchestrator(config)

    try:
        await orchestrator.setup()

        if args.mode == "text":
            await orchestrator.run_text_mode()
        elif args.mode == "audio":
            await orchestrator.run_voice_mode()

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
    finally:
        await orchestrator.shutdown()
        # Force exit — the input() executor thread blocks normal shutdown
        os._exit(0)


def main() -> None:
    """Synchronous entry point."""
    args = parse_args()

    if args.list_providers:
        console = Console()
        providers = list_providers()
        console.print("\n[bold]Available Providers:[/bold]\n")
        for category, names in providers.items():
            console.print(f"  [cyan]{category.upper()}[/cyan]: {', '.join(names)}")
        console.print()
        return

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
