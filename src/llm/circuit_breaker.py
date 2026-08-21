"""Circuit breaker for LLM service resilience.

Implements the circuit breaker pattern to handle llama.cpp server failures:
- CLOSED: Normal operation
- OPEN: Service unavailable, fail fast
- HALF_OPEN: Testing recovery

Auto-recovery after configurable timeout.
"""

import time
import threading
from enum import Enum
from typing import Optional, Callable, Any


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast, service unavailable
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker for LLM API calls.
    
    Tracks consecutive failures and opens the circuit when threshold is exceeded.
    Automatically transitions to half-open state after recovery timeout.
    
    Example usage:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=300)
        
        try:
            result = breaker.call(lambda: llm_service.generate(...))
        except CircuitOpenError:
            print("LLM service unavailable, please wait...")
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 300,  # 5 minutes
        name: str = "llm_service",
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            name: Identifier for logging
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for auto-transition to half-open."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._state = CircuitState.HALF_OPEN
                    print(f"[CIRCUIT] {self.name}: Transitioning to HALF_OPEN (testing recovery)")
            return self._state
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed for recovery attempt."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.recovery_timeout
    
    def is_available(self) -> bool:
        """Check if the service is available (circuit not OPEN)."""
        return self.state != CircuitState.OPEN
    
    def call(self, func: Callable[[], Any], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result from function
            
        Raises:
            CircuitOpenError: If circuit is open (service unavailable)
            Exception: Any exception from the function call
        """
        current_state = self.state
        
        if current_state == CircuitState.OPEN:
            remaining = self.recovery_timeout - (time.time() - (self._last_failure_time or 0))
            raise CircuitOpenError(
                f"LLM service unavailable. Circuit will retry in {remaining:.0f}s"
            )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _on_success(self) -> None:
        """Record successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                print(f"[CIRCUIT] {self.name}: Recovery successful, closing circuit")
            self._failure_count = 0
            self._state = CircuitState.CLOSED
    
    def _on_failure(self, error: Exception) -> None:
        """Record failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Recovery attempt failed, re-open circuit
                self._state = CircuitState.OPEN
                print(f"[CIRCUIT] {self.name}: Recovery failed, circuit OPEN: {error}")
            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN
                print(f"[CIRCUIT] {self.name}: Circuit OPEN after {self._failure_count} failures: {error}")
            else:
                print(f"[CIRCUIT] {self.name}: Failure {self._failure_count}/{self.failure_threshold}: {error}")
    
    def reset(self) -> None:
        """Manually reset the circuit to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._last_failure_time = None
            print(f"[CIRCUIT] {self.name}: Manually reset to CLOSED")
    
    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        with self._lock:
            status = {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
            }
            if self._last_failure_time and self._state == CircuitState.OPEN:
                remaining = self.recovery_timeout - (time.time() - self._last_failure_time)
                status["recovery_in_seconds"] = max(0, remaining)
            return status


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""
    pass


# Global circuit breaker instance for LLM service
_llm_circuit_breaker: Optional[CircuitBreaker] = None


def get_llm_circuit_breaker() -> CircuitBreaker:
    """Get or create the global LLM circuit breaker."""
    global _llm_circuit_breaker
    if _llm_circuit_breaker is None:
        _llm_circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300,  # 5 minutes
            name="llm_server",
        )
    return _llm_circuit_breaker
