#!/usr/bin/env python3
"""Live demo of deferred tool loading in agentknit against Kimi or GLM.

A 30-tool catalog (2 useful, 28 decoys) is declared deferred; the model starts
with only `search_tools`.  Two user turns: "weather in Stockholm + 100 EUR in
SEK", then "and Paris?" (must reuse the revealed tool without searching).

Usage: demo_deferred_tools.py --mode kimi|glm|eager-kimi|eager-glm [--runs N]
Endpoint and key come from inference-db (`kimi-coding`, `zai-coding`).
"""
from __future__ import annotations

import argparse
import sys

import agentknit
from agentknit import Tool, build_tool_spec, register_tools_in_library

TARGETS = {"kimi": ("k3", "kimi-coding"), "glm": ("glm-5.3", "zai-coding")}
TURNS = [("What is the weather in Stockholm, and how much is 100 EUR in SEK?", ["17", "1150"]),
         ("And the weather in Paris?", ["21"])]
NOTES = " Detailed usage notes follow." * 15  # realistic description length
SYSTEM = "You are a helpful assistant. Use tools to get facts; never guess them."


def get_weather(city: str) -> tuple[str, dict]:
    return f'{{"city": "{city}", "temp_c": {21 if "paris" in city.lower() else 17}}}', {"result": "ok"}


def convert_currency(amount: str, from_currency: str, to_currency: str) -> tuple[str, dict]:
    return f'{{"result": 1150, "currency": "{to_currency}"}}', {"result": "ok"}


def decoy(query: str) -> tuple[str, dict]:
    return "ERROR: this tool is not relevant here", {"result": "error", "ok": False}


DECOYS = ["get_stock_price: Latest stock price for a ticker", "translate_text: Translate text between languages",
          "send_email: Send an email", "create_calendar_event: Create a calendar event",
          "search_flights: Search flights between airports", "book_hotel: Book a hotel room",
          "get_news: Latest news headlines on a topic", "run_sql: Run a SQL query on the warehouse",
          "geocode_address: Convert an address to coordinates", "get_air_quality: Air quality index for a city",
          "get_sunrise_sunset: Sunrise and sunset times for a city", "get_exchange_holidays: Bank holidays",
          "create_ticket: Open a support ticket", "list_files: List files in a folder", "read_file2: Read a file",
          "summarize_url: Summarize a web page", "get_traffic: Traffic conditions on a route",
          "get_tide: Tide times at a port", "get_crypto_price: Price of a cryptocurrency",
          "calc_mortgage: Monthly mortgage payment", "get_timezone: Time zone of a city",
          "spell_check: Spell-check text", "get_recipe: Find a recipe", "track_package: Track a parcel",
          "get_movie_times: Cinema showtimes", "get_pollen: Pollen forecast for a city",
          "get_uv_index: UV index for a city", "get_inflation: Inflation rate of a country"]


def catalog(deferred: bool) -> list[Tool]:
    tools = [Tool("get_weather", "Get current weather (temperature, conditions) for a city." + NOTES,
                  get_weather, deferred=deferred),
             Tool("convert_currency", "Convert an amount of money between two currencies." + NOTES,
                  convert_currency, deferred=deferred)]
    for line in DECOYS:
        name, desc = line.split(": ")
        tools.append(Tool(name, desc + "." + NOTES, decoy, deferred=deferred))
    return tools


def run(mode: str, compact: bool = False) -> tuple[list[bool], list[str], list[tuple[int, int]]]:
    provider = mode.removeprefix("eager-")
    model, entry = TARGETS[provider]
    tools = catalog(deferred=not mode.startswith("eager"))
    schema_tools, dispatch = build_tool_spec(tools)
    register_tools_in_library(tools)
    schema = agentknit.load_specification(model, inference_db=entry)
    # Replace the spec's default toolset (tool_specs wins over inferred_tool_schema).
    schema.update({"tool_specs": schema_tools, "tool_dispatch": dispatch})
    schema.pop("tools", None)
    calls: list[str] = []
    usage: list[tuple[int, int]] = []

    def on_event(kind: str, data: dict) -> None:
        if kind == "tool_call":
            calls.append(str(data["name"]))
        elif kind == "usage" and data.get("purpose") is None:
            usage.append((int(data.get("prompt") or 0), int(data.get("cached") or 0)))
        elif kind == "error":
            print("ERROR", str(data.get("text"))[:300], flush=True)

    session = agentknit.init_session(schema, non_interactive=True, bare=True, on_event=on_event,
                                     system_prompt_supplement=SYSTEM,
                                     strict_cache_proof=False,
                                     # Deferred modes come from inference-db's tool_loading quirk.
                                     tool_loading="eager" if mode.startswith("eager") else None)
    if not mode.startswith("eager"):
        print(f"tool_loading from inference-db: {schema.get('tool_loading')} -> {session.get('tool_loading')}",
              flush=True)
    client = agentknit.create_client(schema)
    ok = []
    turn_calls = []
    for turn, (task, expect) in enumerate(TURNS):
        if turn and compact:
            # Summarize the whole history: revealed tools must survive it.
            session["compaction_keep_last_turns"] = 0
            ok_compact = agentknit.compact_session(client, model, session)
            print(f"compacted={ok_compact} tools_sent={len(session['tools'])} "
                  f"deferred={sum(bool((t.get('function') or t).get('defer_loading')) for t in session['tools'])}",
                  flush=True)
        calls.clear()
        result = agentknit.run_turn(client, model, session, task)
        reply = (result.final_reply or "").replace(",", "").replace(" ", "")
        ok.append(all(v in reply for v in expect))
        turn_calls.append(">".join(calls))
    return ok, turn_calls, usage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["kimi", "glm", "eager-kimi", "eager-glm"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--compact", action="store_true", help="compact the whole history between the turns")
    a = ap.parse_args()
    for i in range(a.runs):
        ok, calls, usage = run(a.mode, a.compact)
        turns = " || ".join(f"{'✅' if o else '❌'} {c}" for o, c in zip(ok, calls, strict=True))
        print(f"RESULT {a.mode} run{i} | {turns} | prompt/cached: "
              + " ".join(f"{p}/{c}" for p, c in usage), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
