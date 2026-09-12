# test_msf.py
import asyncio
import sys
sys.path.insert(0, '.')

from app.exploit.metasploit_client import get_msf_client, close_msf_client


async def test():
    client = get_msf_client()
    try:
        await client.connect()
        print('✅ Connected to msfrpcd!')

        modules = await client.module_search('eternalblue')
        print(f'Found {len(modules)} modules for eternalblue')
        for m in modules[:3]:
            print(f'  - {m["type"]}/{m["name"]}: {m["description"][:60]}')
    finally:
        await close_msf_client()


if __name__ == '__main__':
    asyncio.run(test())