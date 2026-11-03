import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from jarvis.chat import ChatSession

async def main():
    s = ChatSession()
    s.cfg = s.load_config()
    ready, reason = s.validate_mode("online")
    print("validate_mode online:", ready, reason)
    s.setup_providers("online")
    print("chain providers:", [p.name for p in s.provider_chain.providers])
    chunks = []
    async for chunk in s.get_response_stream("hey"):
        print("CHUNK:", repr(chunk))
        chunks.append(chunk)
    print("FULL:", "".join(chunks))

if __name__ == "__main__":
    asyncio.run(main())

