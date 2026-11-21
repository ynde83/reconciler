import asyncio
import os

import nats
from nats.js.api import StreamConfig

async def main():
    nc = await nats.connect(os.environ.get("NATS_URL", "nats://localhost:4222"))
    js = nc.jetstream()

    try:
        info = await js.stream_info("MLOPS")
        print("stream exists:", info.config.subjects)
    except nats.js.errors.NotFoundError:
        await js.add_stream(StreamConfig(name="MLOPS", subjects=["mlops.>"], duplicate_window=120))
        print("stream created: MLOPS")
    await nc.close()

asyncio.run(main())