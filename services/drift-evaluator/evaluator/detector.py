class DriftDetector:
    def __init__(self, cooldown_sec: float):
        self.cooldown_sec = cooldown_sec
        self.state = "WARMUP"
        self.cooldown_until = 0.0

    def update(self, is_drift: bool, now: float) -> bool:
        if now < self.cooldown_until:
            self.state = "COOLDOWN"
            return False

        if is_drift:
            if self.state == "SUSPECTED":
                self.state = "CONFIRMED"
                self.cooldown_until = now + self.cooldown_sec
                return True
            self.state = "SUSPECTED"
            return False

        self.state = "OK"
        return False
    def cooldown_remaining(self, now: float) -> float:
        return max(0.0, self.cooldown_until - now)