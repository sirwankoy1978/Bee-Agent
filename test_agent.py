"""Functional smoke test — run the agent end-to-end (no Telegram)."""
import asyncio
import sys

import bot as b

async def main():
    print("Running a real agent call (short test)...")
    resp = await b.agent.arun(
        "What is 2+2? Answer with just the number.",
        stream=False,
    )
    print(f"\nStatus: {resp.status}")
    print(f"Content: {resp.content}")
    if resp.status == "ERROR":
        print("ERROR DETAIL:", resp)
        sys.exit(1)
    assert resp.content and "4" in resp.content, "Unexpected answer"
    print("\n✅ Agent works correctly.")

if __name__ == "__main__":
    asyncio.run(main())
