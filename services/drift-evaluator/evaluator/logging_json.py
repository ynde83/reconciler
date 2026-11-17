import json
import logging
import sys
from datetime import datetime, timezone

SERVICE = "model-api"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        data = {
            "time": datetime.fromtimestamp(record.created, timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "service": SERVICE,
        }

        if (
            record.name == "uvicorn.access"
            and isinstance(record.args, tuple)
            and len(record.args) == 5
        ):
            client, method, path, _, status = record.args
            data.update(
                msg="http_request",
                method=method,
                path=path,
                status=status,
                client=client,
            )
        else:
            data["msg"] = record.getMessage()

        data.update(getattr(record, "fields", {}))

        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str)


def setup_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


logger = logging.getLogger(SERVICE)


def log(level: str, msg: str, **fields):
    logger.info(getattr(logging, level), msg, extra={"fields": fields})
