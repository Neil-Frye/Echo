# Python Style Guide

## 📋 Table of Contents

* [Introduction](#introduction)
* [General Principles](#general-principles)
* [Code Layout](#code-layout)

  * [Project Structure](#project-structure)
  * [Imports](#imports)
  * [Line Length and Indentation](#line-length-and-indentation)
* [Naming Conventions](#naming-conventions)

  * [Variables and Functions](#variables-and-functions)
  * [Classes](#classes)
  * [Constants](#constants)
* [Type Hints](#type-hints)

  * [Basic Type Hints](#basic-type-hints)
  * [Advanced Type Hints](#advanced-type-hints)
  * [Type Aliases](#type-aliases)
* [Functions and Methods](#functions-and-methods)

  * [Function Design](#function-design)
  * [Decorators](#decorators)
* [Classes](#classes-1)

  * [Class Design](#class-design)
* [Error Handling](#error-handling)

  * [Exception Classes](#exception-classes)
  * [Exception Handling](#exception-handling)
* [Async Programming](#async-programming)

  * [Async Functions](#async-functions)
* [AI/ML Guidelines](#aiml-guidelines)

  * [Model Implementation](#model-implementation)
  * [Training Loop](#training-loop)
* [Testing](#testing)
* [Performance](#performance)
* [Security](#security)

---

## 🎯 Introduction

This style guide defines Python coding standards for EthernalEcho's AI and backend services. We follow PEP 8 as our foundation, with additional guidelines specific to our domain.

---

## 🏗️ General Principles

1. **Clarity over Cleverness**: Write code that's easy to understand.
2. **Type Safety**: Use type hints everywhere.
3. **Explicit over Implicit**: Follow the Zen of Python.
4. **Documentation**: Every public API must be documented.
5. **Testability**: Design code that's easy to test.

---

## 📐 Code Layout

### Project Structure

```
services/
├── ai/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── voice_synthesis.py
│   │   └── personality_model.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── audio_processor.py
│   │   └── text_processor.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_utils.py
│   └── tests/
│       ├── __init__.py
│       ├── test_voice_synthesis.py
│       └── test_personality_model.py
├── api/
│   ├── __init__.py
│   ├── routers/
│   ├── dependencies/
│   └── middleware/
└── shared/
    ├── __init__.py
    ├── config.py
    └── logging.py
```

* **services/ai**: Contains AI models, processors, utilities, and tests.
* **services/api**: FastAPI routers, dependencies, and middleware.
* **shared/**: Common configurations and logging utilities.

---

### Imports

```python
# ✅ Good - Organized imports

# Standard library imports
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

# Related third-party imports
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Local application imports
from app.models import VoiceModel, PersonalityModel
from app.processors import AudioProcessor
from app.utils import load_checkpoint, save_checkpoint


# ❌ Bad - Disorganized imports

from app.models import *  # Avoid wildcard imports
import torch
from datetime import datetime
import os
from app.utils import load_checkpoint
import numpy as np
```

* **Group imports** by type: standard library, third-party, then local.
* **Avoid** wildcard imports like `from module import *`.
* **Alphabetize** within each group if there are many imports.

---

### Line Length and Indentation

```python
# ✅ Good - Proper line breaking

def process_audio_sample(
    audio_data: np.ndarray,
    sample_rate: int = 16000,
    normalize: bool = True,
    remove_silence: bool = True,
    min_silence_duration: float = 0.1
) -> Dict[str, Any]:
    """Process raw audio data and extract features.
    
    Args:
        audio_data: Raw audio samples as numpy array
        sample_rate: Audio sample rate in Hz
        normalize: Whether to normalize audio amplitude
        remove_silence: Whether to remove silent segments
        min_silence_duration: Minimum duration of silence to remove
        
    Returns:
        Dictionary containing processed audio and metadata
    """
    if len(audio_data) == 0:
        raise ValueError("Audio data cannot be empty")
    
    # Process audio...
    return processed_data


# ❌ Bad - Poor formatting

def process_audio_sample(audio_data: np.ndarray, sample_rate: int = 16000, normalize: bool = True, remove_silence: bool = True, min_silence_duration: float = 0.1) -> Dict[str, Any]:
    """Process raw audio data and extract features."""
    if len(audio_data) == 0: raise ValueError("Audio data cannot be empty")
    # Process audio...
    return processed_data
```

* **Limit** lines to 88 characters (PEP 8).
* Indent continuation lines with **4 spaces** and break after commas in parameter lists.
* **Avoid** putting multiple statements on one line (e.g., `if ...: raise ...`).

---

## 📝 Naming Conventions

### Variables and Functions

```python
# ✅ Good naming

user_profile = await fetch_user_profile(user_id)
is_recording = False
max_retry_attempts = 3
audio_sample_rate = 16000

def calculate_voice_quality(samples: List[VoiceSample]) -> float:
    """Calculate average voice quality from samples."""
    if not samples:
        return 0.0
    return sum(s.quality for s in samples) / len(samples)


# ❌ Bad naming

prof = await get_prof(id)  # Too abbreviated
recording = False          # Ambiguous boolean name
MAX_RETRY = 3              # Should be lowercase (unless it's a constant)
audioSampleRate = 16000    # Should be snake_case

def calc(s: List[Any]) -> float:  # Unclear function name
    return sum(s.q for s in s) / len(s)
```

* Use **snake\_case** for variables and function names.
* Make names **descriptive** and avoid unnecessary abbreviations.
* Boolean flags should start with `is_`, `has_`, `should_`, etc.

---

### Classes

```python
# ✅ Good - Clear class names

class VoiceEncoder(nn.Module):
    """Encodes voice samples into embeddings."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        return self.projection(encoded[:, -1, :])


class AudioProcessor:
    """Processes raw audio data."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._cache: Dict[str, Any] = {}
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio data."""
        return self._apply_filters(audio)
    
    def _apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio filters (private method)."""
        # Implementation
        pass


# ❌ Bad - Poor class design

class voice_encoder(nn.Module):  # Should be PascalCase
    def __init__(self):
        super().__init__()
        # Missing parameters
    
    def process(self, x):  # Missing type hints
        # Implementation
        pass
```

* Class names should be **PascalCase** (e.g., `VoiceEncoder`).
* Add **docstrings** at the class level.
* Use type hints for method parameters and return types.

---

### Constants

```python
# ✅ Good - Clear constants

# Audio processing constants
DEFAULT_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION = 300  # seconds
SUPPORTED_AUDIO_FORMATS = ("mp3", "wav", "webm", "ogg")

# Model configuration
MODEL_CONFIGS = {
    "voice_encoder": {
        "hidden_dim": 512,
        "num_layers": 6,
        "dropout": 0.1
    },
    "personality_model": {
        "embedding_dim": 768,
        "num_heads": 12,
        "max_length": 2048
    }
}

# API configuration
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"


# ❌ Bad - Poor constant naming

sample_rate = 16000          # Should be uppercase
MAXDURATION = 300            # Should be snake_case
supported_formats = ["mp3"]  # Should be uppercase and tuple
```

* Constants should be in **UPPER\_SNAKE\_CASE**.
* Tuples preferred over lists for immutable groups (e.g., `("mp3", "wav")`).

---

## 🔧 Type Hints

### Basic Type Hints

```python
# ✅ Good - Comprehensive type hints

from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch


def process_voice_samples(
    samples: List[np.ndarray],
    sample_rate: int,
    target_length: Optional[int] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Process multiple voice samples.
    
    Args:
        samples: List of audio samples as numpy arrays
        sample_rate: Sample rate in Hz
        target_length: Target length for padding/truncation
        
    Returns:
        Tuple of (processed tensor, metadata dict)
    """
    processed_samples = []
    metadata = {
        "total_duration": 0.0,
        "average_energy": 0.0,
        "sample_count": len(samples)
    }
    
    for sample in samples:
        processed = _process_single_sample(sample, sample_rate, target_length)
        processed_samples.append(processed)
        metadata["total_duration"] += len(sample) / sample_rate
    
    return torch.stack(processed_samples), metadata


# ❌ Bad - Missing or incorrect type hints

def process_voice_samples(samples, sample_rate, target_length=None):
    """Process multiple voice samples."""
    processed_samples = []
    metadata = {}
    
    for sample in samples:
        processed = _process_single_sample(sample, sample_rate, target_length)
        processed_samples.append(processed)
    
    return processed_samples, metadata
```

* Always annotate parameters and return types explicitly.
* Use `Optional[...]` instead of defaulting to `None` without a hint.

---

### Advanced Type Hints

```python
# ✅ Good - Advanced type usage

from typing import TypeVar, Generic, Protocol, Literal, TypedDict, Callable
from typing_extensions import ParamSpec
import numpy.typing as npt
import torch

# Type variables
T = TypeVar("T", bound=np.generic)
P = ParamSpec("P")

# Typed dictionary
class VoiceMetadata(TypedDict):
    duration: float
    sample_rate: int
    channels: int
    format: Literal["mp3", "wav", "webm"]
    quality: float

# Protocol for model interface
class VoiceModel(Protocol):
    def encode(self, audio: npt.NDArray[np.float32]) -> torch.Tensor: ...
    def decode(self, embedding: torch.Tensor) -> npt.NDArray[np.float32]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

# Generic class
class ModelCache(Generic[T]):
    def __init__(self) -> None:
        self._cache: Dict[str, T] = {}
    
    def get(self, key: str) -> Optional[T]:
        return self._cache.get(key)
    
    def set(self, key: str, value: T) -> None:
        self._cache[key] = value

# Callable type hint
ProcessingFunction = Callable[[npt.NDArray[np.float32], int], npt.NDArray[np.float32]]


def apply_processing(
    audio: npt.NDArray[np.float32],
    processors: List[ProcessingFunction]
) -> npt.NDArray[np.float32]:
    """Apply a chain of processing functions to audio."""
    result = audio
    for processor in processors:
        result = processor(result, 16000)
    return result
```

* Use **`Protocol`** to define interfaces.
* Use **`TypedDict`** for structured dictionaries.
* Use **`ParamSpec`** and **`TypeVar`** for generic, flexible function signatures.

---

## 🎯 Functions and Methods

### Function Design

```python
# ✅ Good - Well-designed functions

def create_voice_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
    pretrained: bool = True,
    device: str = "cuda"
) -> VoiceModel:
    """Create a voice model instance.
    
    Args:
        model_type: Type of model to create ('synthesizer', 'analyzer')
        config: Model configuration override
        pretrained: Whether to load pretrained weights
        device: Device to load model on
        
    Returns:
        Initialized voice model
        
    Raises:
        ValueError: If model_type is not supported
        RuntimeError: If pretrained weights cannot be loaded
    """
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model type: {model_type}. "
            f"Choose from: {', '.join(SUPPORTED_MODELS)}"
        )
    
    model_config = DEFAULT_CONFIGS[model_type].copy()
    if config:
        model_config.update(config)
    
    model = MODEL_CLASSES[model_type](**model_config)
    
    if pretrained:
        weights_path = get_pretrained_weights_path(model_type)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    
    return model.to(device)


# ❌ Bad - Poor function design

def create_model(type, config={}):  # Mutable default argument!
    """Create model."""
    model = MODEL_CLASSES[type](**config)  # No validation
    return model  # No error handling
```

* **Validate** inputs early (e.g., supported model types).
* **Avoid** mutable default arguments (e.g., `config={}`) — use `None` and then set a new dict inside.
* Use **docstrings** that describe `Args:`, `Returns:`, and `Raises:`.

---

### Decorators

```python
# ✅ Good - Useful decorators

import functools
import time
import logging
from typing import Callable, TypeVar, Any

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

def timer(func: F) -> F:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            logger.info(f"{func.__name__} took {elapsed:.3f} seconds")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f} seconds: {e}"
            )
            raise
    return wrapper


def validate_audio_input(func: F) -> F:
    """Decorator to validate audio input parameters."""
    @functools.wraps(func)
    def wrapper(audio: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        if not isinstance(audio, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(audio)}")
        if audio.ndim not in (1, 2):
            raise ValueError(f"Audio must be 1D or 2D, got {audio.ndim}D")
        if len(audio) == 0:
            raise ValueError("Audio array is empty")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or infinite values")
        return func(audio, *args, **kwargs)
    return wrapper


@timer
@validate_audio_input
def process_audio_chunk(
    audio: np.ndarray,
    sample_rate: int = 16000
) -> np.ndarray:
    """Process an audio chunk with validation and timing."""
    # Processing logic
    return processed_audio
```

* Combine **logging** and **timing** logic via wrappers.
* Use **type checks** and **value checks** in a decorator rather than duplicating validation in each function.

---

## 🏛️ Classes

### Class Design

```python
# ✅ Good - Well-designed class

from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np


@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    normalize: bool = True
    remove_silence: bool = True
    min_silence_duration: float = 0.1
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("Channels must be 1 (mono) or 2 (stereo)")
        if self.bit_depth not in (16, 24, 32):
            raise ValueError("Bit depth must be 16, 24, or 32")


class BaseAudioProcessor(ABC):
    """Abstract base class for audio processors."""
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """Initialize processor with configuration."""
        self.config = config or AudioConfig()
        self._cache: Dict[str, Any] = {}
        self._is_initialized = False
    
    @abstractmethod
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset processor state. Must be implemented by subclasses."""
        pass
    
    def __enter__(self) -> BaseAudioProcessor:
        """Context manager entry."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.cleanup()
    
    def initialize(self) -> None:
        """Initialize processor resources."""
        if not self._is_initialized:
            self._setup_filters()
            self._is_initialized = True
    
    def cleanup(self) -> None:
        """Clean up processor resources."""
        self._cache.clear()
        self._is_initialized = False
    
    def _setup_filters(self) -> None:
        """Set up audio filters (private)."""
        # Implementation details for initializing filters
        pass


class VoiceEnhancer(BaseAudioProcessor):
    """Enhances voice quality in audio recordings."""
    
    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        noise_reduction_strength: float = 0.5
    ):
        """Initialize voice enhancer.
        
        Args:
            config: Audio processing configuration
            noise_reduction_strength: Strength of noise reduction (0–1)
        """
        super().__init__(config)
        self.noise_reduction_strength = noise_reduction_strength
        self.noise_profile: Optional[np.ndarray] = None
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Process audio to enhance voice quality.
        
        Args:
            audio: Input audio array
            
        Returns:
            Enhanced audio array
        """
        if not self._is_initialized:
            self.initialize()
        
        # Apply processing chain
        audio = self._normalize(audio)
        audio = self._reduce_noise(audio)
        audio = self._enhance_voice(audio)
        
        if self.config.remove_silence:
            audio = self._remove_silence(audio)
        
        return audio
    
    def reset(self) -> None:
        """Reset enhancer state."""
        self.noise_profile = None
        self._cache.clear()
    
    def learn_noise_profile(self, noise_sample: np.ndarray) -> None:
        """Learn noise profile from a sample.
        
        Args:
            noise_sample: Audio sample containing only noise
        """
        self.noise_profile = self._compute_spectrum(noise_sample)
    
    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude."""
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val * 0.95
        return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Reduce noise using spectral subtraction."""
        if self.noise_profile is None:
            return audio
        
        spectrum = self._compute_spectrum(audio)
        clean_spectrum = spectrum - self.noise_reduction_strength * self.noise_profile
        clean_spectrum = np.maximum(clean_spectrum, 0)
        
        return self._inverse_spectrum(clean_spectrum)
    
    def _enhance_voice(self, audio: np.ndarray) -> np.ndarray:
        """Enhance voice frequencies."""
        # Implementation details (e.g., equalization)
        return audio
    
    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove silence from audio."""
        # Implementation details (e.g., VAD-based trimming)
        return audio
    
    def _compute_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """Compute frequency spectrum."""
        return np.fft.rfft(audio)
    
    def _inverse_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute inverse spectrum."""
        return np.fft.irfft(spectrum)
```

---

## 🚨 Error Handling

### Exception Classes

```python
# ✅ Good - Custom exception hierarchy

class EthernalEchoError(Exception):
    """Base exception for all EthernalEcho errors."""
    pass


class AudioProcessingError(EthernalEchoError):
    """Raised when audio processing fails."""
    
    def __init__(
        self,
        message: str,
        audio_format: Optional[str] = None,
        sample_rate: Optional[int] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.error_code = error_code


class ModelNotFoundError(EthernalEchoError):
    """Raised when a model cannot be found."""
    
    def __init__(self, model_id: str, model_type: str):
        super().__init__(f"{model_type} model not found: {model_id}")
        self.model_id = model_id
        self.model_type = model_type


class VoiceQualityError(AudioProcessingError):
    """Raised when voice quality is insufficient."""
    
    def __init__(
        self,
        quality_score: float,
        min_required: float,
        details: Optional[Dict[str, Any]] = None
    ):
        message = (
            f"Voice quality {quality_score:.2f} is below "
            f"minimum required {min_required:.2f}"
        )
        super().__init__(message)
        self.quality_score = quality_score
        self.min_required = min_required
        self.details = details or {}
```

* Define a **base exception** (`EthernalEchoError`) to catch unexpected errors across the codebase.
* Subclass with **contextual information** for different failure cases (e.g., `AudioProcessingError`, `VoiceQualityError`).

---

### Exception Handling

```python
# ✅ Good - Proper exception handling

import logging
from typing import Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


async def process_voice_upload(
    file_path: Union[str, Path],
    user_id: str
) -> Dict[str, Any]:
    """Process uploaded voice file.
    
    Args:
        file_path: Path to uploaded file
        user_id: ID of user uploading the file
        
    Returns:
        Processing results including quality metrics
        
    Raises:
        AudioProcessingError: If audio processing fails
        VoiceQualityError: If voice quality is insufficient
    """
    file_path = Path(file_path)
    
    # Validate file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    try:
        # Load and validate audio
        audio_data, sample_rate = load_audio(file_path)
        logger.info(
            f"Loaded audio file: {file_path.name}, "
            f"duration: {len(audio_data) / sample_rate:.2f}s"
        )
        
        # Process audio
        try:
            features = extract_voice_features(audio_data, sample_rate)
            quality_score = calculate_quality_score(features)
            
            # Check quality threshold
            if quality_score < MIN_QUALITY_THRESHOLD:
                raise VoiceQualityError(
                    quality_score=quality_score,
                    min_required=MIN_QUALITY_THRESHOLD,
                    details={
                        "low_snr": features.get("snr", 0) < 10,
                        "clipping": features.get("clipping_ratio", 0) > 0.01,
                        "silence_ratio": features.get("silence_ratio", 0) > 0.5
                    }
                )
            
            # Save processed data
            processed_path = await save_processed_audio(
                audio_data,
                sample_rate,
                user_id
            )
            
            return {
                "status": "success",
                "file_path": str(processed_path),
                "quality_score": quality_score,
                "features": features,
                "duration": len(audio_data) / sample_rate
            }
            
        except VoiceQualityError:
            # Re-raise quality errors
            raise
        except Exception as e:
            # Wrap other processing errors
            logger.error(f"Audio processing failed: {e}", exc_info=True)
            raise AudioProcessingError(
                f"Failed to process audio: {str(e)}",
                audio_format=file_path.suffix[1:],
                sample_rate=sample_rate
            ) from e
            
    except (FileNotFoundError, AudioProcessingError, VoiceQualityError):
        # Re-raise expected errors
        raise
    except Exception as e:
        # Log unexpected errors
        logger.error(
            f"Unexpected error processing voice upload for user {user_id}: {e}",
            exc_info=True
        )
        raise EthernalEchoError(
            "An unexpected error occurred during voice processing"
        ) from e
    finally:
        # Cleanup temporary files
        if file_path.exists() and file_path.parent.name == "temp":
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


# ❌ Bad - Poor exception handling

def process_voice_upload(file_path, user_id):
    """Process uploaded voice file."""
    try:
        audio_data = load_audio(file_path)
        features = extract_voice_features(audio_data)
        return {"status": "success", "features": features}
    except:  # Bare except (discouraged)
        print("Error occurred")  # Inadequate logging
        return None  # Swallows error without context
```

* Catch **specific exceptions** instead of using bare `except:`.
* Use **logging** rather than `print`.
* **Wrap** unexpected exceptions in a higher-level exception to preserve context.

---

## ⚡ Async Programming

### Async Functions

```python
# ✅ Good - Proper async implementation

import asyncio
from typing import List, Optional, AsyncIterator
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor


class AsyncVoiceProcessor:
    """Asynchronous voice processing service."""
    
    def __init__(self, max_workers: int = 4):
        self.session: Optional[aiohttp.ClientSession] = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def __aenter__(self) -> "AsyncVoiceProcessor":
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
    
    async def process_voice_batch(
        self,
        file_paths: List[Path]
    ) -> List[Dict[str, Any]]:
        """Process multiple voice files concurrently.
        
        Args:
            file_paths: List of audio file paths
            
        Returns:
            List of processing results
        """
        tasks = [self.process_voice_file(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for path, result in zip(file_paths, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {path}: {result}")
                processed_results.append({
                    "file": str(path),
                    "status": "error",
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def process_voice_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single voice file asynchronously.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Processing results
        """
        # Read file asynchronously
        async with aiofiles.open(file_path, "rb") as f:
            audio_data = await f.read()
        
        # CPU-intensive processing in thread pool
        features = await asyncio.get_event_loop().run_in_executor(
            self.executor,
            self._extract_features_sync,
            audio_data
        )
        
        # Upload results asynchronously
        if self.session:
            upload_result = await self._upload_features(features)
            
        return {
            "file": str(file_path),
            "status": "success",
            "features": features,
            "uploaded": upload_result
        }
    
    def _extract_features_sync(self, audio_data: bytes) -> Dict[str, float]:
        """CPU-intensive feature extraction (sync)."""
        audio = np.frombuffer(audio_data, dtype=np.int16)
        return {
            "duration": len(audio) / 16000,
            "energy": float(np.mean(np.abs(audio))),
            "zero_crossings": float(np.sum(np.diff(np.sign(audio)) != 0))
        }
    
    async def _upload_features(
        self,
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Upload features to API."""
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        async with self.session.post(
            "https://api.ethernalecho.com/features",
            json=features
        ) as response:
            return await response.json()
    
    async def stream_voice_samples(
        self,
        user_id: str
    ) -> AsyncIterator[VoiceSample]:
        """Stream voice samples for a user.
        
        Args:
            user_id: User ID to stream samples for
            
        Yields:
            Voice samples as they become available
        """
        if not self.session:
            raise RuntimeError("Session not initialized")
            
        async with self.session.get(
            f"https://api.ethernalecho.com/users/{user_id}/samples/stream"
        ) as response:
            async for line in response.content:
                if line:
                    sample_data = json.loads(line)
                    yield VoiceSample(**sample_data)


# Usage example

async def main():
    """Example usage of async voice processor."""
    async with AsyncVoiceProcessor() as processor:
        # Process files concurrently
        results = await processor.process_voice_batch([
            Path("sample1.wav"),
            Path("sample2.wav"),
            Path("sample3.wav")
        ])
        
        # Stream samples
        async for sample in processor.stream_voice_samples("user123"):
            print(f"Received sample: {sample.id}")


# ❌ Bad - Poor async implementation

async def process_files(files):
    """Process files."""
    results = []
    for file in files:  # Not concurrent!
        result = await process_file(file)
        results.append(result)
    return results

async def process_file(file):
    """Process single file."""
    # Blocking I/O in async function
    with open(file, "rb") as f:
        data = f.read()  # Should use aiofiles
    
    # CPU-intensive work in async function
    features = extract_features(data)  # Should use executor
    
    return features
```

* **Use `aiofiles`** for non-blocking file I/O.
* Offload CPU-bound work to a **ThreadPoolExecutor**.
* Use **`asyncio.gather`** to run tasks concurrently.

---

## 🤖 AI/ML Guidelines

### Model Implementation

```python
# ✅ Good - Well-structured AI model

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for voice synthesis model."""
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_length: int = 5000
    vocab_size: int = 256
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "max_length": self.max_length,
            "vocab_size": self.vocab_size
        }


class VoiceSynthesisModel(nn.Module):
    """Transformer-based voice synthesis model."""
    
    def __init__(self, config: ModelConfig):
        """Initialize model with configuration."""
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return hidden states
            
        Returns:
            If return_hidden_states is False: logits tensor [batch_size, seq_len, vocab_size]  
            If True: (logits, hidden_states)
        """
        batch_size, seq_len = input_ids.shape
        
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        hidden_states = token_embeds + position_embeds
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        
        if attention_mask is not None:
            attention_mask = ~attention_mask.bool()  # Masked tokens are True
        
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=attention_mask
        )
        
        logits = self.output_projection(hidden_states)
        
        if return_hidden_states:
            return logits, hidden_states
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[torch.Tensor] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> torch.Tensor:
        """Generate audio tokens autoregressively.
        
        Args:
            prompt: Optional prompt tokens [batch_size, prompt_len]
            max_length: Maximum length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            
        Returns:
            Generated token sequences [batch_size, <= max_length]
        """
        self.eval()
        device = next(self.parameters()).device
        
        if prompt is not None:
            generated = prompt.to(device)
        else:
            generated = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        for _ in range(max_length - generated.shape[1]):
            logits = self(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            filtered_logits = self._top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p
            )
            
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == self.config.vocab_size - 1:
                break
        
        return generated
    
    @staticmethod
    def _top_k_top_p_filtering(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
        filter_value: float = -float("inf")
    ) -> torch.Tensor:
        """Filter logits using top-k and top-p filtering.
        
        Args:
            logits: Logits tensor [batch_size, vocab_size]
            top_k: Keep only top_k tokens
            top_p: Keep tokens with cumulative prob >= top_p
            filter_value: Value to assign to filtered logits
            
        Returns:
            Filtered logits [batch_size, vocab_size]
        """
        if top_k > 0:
            topk_values = torch.topk(logits, top_k)[0][..., -1, None]
            indices_to_remove = logits < topk_values
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value
        
        return logits
    
    def save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict()
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    @classmethod
    def from_checkpoint(cls, path: Path) -> "VoiceSynthesisModel":
        """Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location="cpu")
        config = ModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")
        return model
```

* Initialize **embeddings** and **transformer layers** in `__init__`.
* Separate **weight initialization** into a private method.
* Provide both `forward` and `generate` methods for inference.
* Keep sampling logic modular (`_top_k_top_p_filtering`).

---

### Training Loop

```python
# ✅ Good - Well-structured training loop

from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import wandb
from tqdm import tqdm
from pathlib import Path


class VoiceTrainer:
    """Trainer for voice synthesis models."""
    
    def __init__(
        self,
        model: VoiceSynthesisModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        gradient_clip: float = 1.0,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        use_wandb: bool = True
    ):
        """Initialize trainer with configuration."""
        self.model = model.to(device)
        self.device = device
        self.gradient_clip = gradient_clip
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=warmup_steps,
            T_mult=2
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        
        # Initialize Weights & Biases
        if use_wandb:
            wandb.init(project="ethernalecho-voice", config={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "model_config": model.config.to_dict()
            })
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
            self.global_step += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/gradient_norm": self._get_gradient_norm()
                }, step=self.global_step)
        
        epoch_loss = total_loss / total_tokens
        epoch_perplexity = torch.exp(torch.tensor(epoch_loss)).item()
        return {"loss": epoch_loss, "perplexity": epoch_perplexity}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model on validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            logits = self.model(input_ids, attention_mask)
            
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
        
        val_loss = total_loss / total_tokens
        val_perplexity = torch.exp(torch.tensor(val_loss)).item()
        
        if self.use_wandb:
            wandb.log({
                "val/loss": val_loss,
                "val/perplexity": val_perplexity
            }, step=self.global_step)
        
        return {"loss": val_loss, "perplexity": val_perplexity}
    
    def train(
        self,
        num_epochs: int,
        validate_every: int = 1,
        save_every: int = 1
    ) -> None:
        """Train model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            validate_every: Validate every N epochs
            save_every: Save checkpoint every N epochs
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"Epoch {epoch}/{num_epochs}")
            
            train_metrics = self.train_epoch()
            logger.info(f"Train metrics: {train_metrics}")
            
            if validate_every > 0 and (epoch % validate_every == 0):
                val_metrics = self.validate()
                logger.info(f"Validation metrics: {val_metrics}")
                
                if val_metrics.get("loss", float("inf")) < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.save_checkpoint(self.checkpoint_dir / "best_model.pt")
            
            if save_every > 0 and (epoch % save_every == 0):
                self.save_checkpoint(self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
        
        logger.info("Training completed")
    
    def _get_gradient_norm(self) -> float:
        """Calculate gradient norm for monitoring."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
```

* Log **training** and **validation** metrics consistently.
* Use **`torch.no_grad()`** during validation to prevent gradient computation.
* **Save checkpoints** conditionally based on performance.

---

## 🧪 Testing

### Test Structure

```python
# ✅ Good - Well-structured tests

import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

# Test fixtures

@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    duration = 1.0  # seconds
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    # Generate sine wave with noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    return audio.astype(np.float32)


@pytest.fixture
def audio_config():
    """Create test audio configuration."""
    return AudioConfig(
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        normalize=True
    )


@pytest.fixture
def voice_model():
    """Create test voice model."""
    config = ModelConfig(hidden_dim=128, num_layers=2, num_heads=4, max_length=50, vocab_size=256)
    return VoiceSynthesisModel(config)


# Test suites

class TestAudioProcessor:
    """Test suite for audio processor."""
    
    def test_initialization(self, audio_config):
        """Test processor initialization."""
        processor = VoiceEnhancer(config=audio_config)
        assert processor.config == audio_config
        assert processor.noise_reduction_strength == 0.5
        assert processor.noise_profile is None
        assert not processor._is_initialized
    
    def test_normalize_audio(self, sample_audio):
        """Test audio normalization."""
        audio = np.array([0.5, -1.0, 0.8, -0.3], dtype=np.float32)
        normalized = VoiceEnhancer._normalize(audio)
        assert np.abs(normalized).max() == pytest.approx(0.95, rel=1e-5)
        assert normalized.shape == audio.shape
        assert np.allclose(normalized / normalized.max(), audio / audio.max())
    
    def test_process_empty_audio(self, audio_config):
        """Test processing empty audio raises error."""
        processor = VoiceEnhancer(config=audio_config)
        empty_audio = np.array([], dtype=np.float32)
        with pytest.raises(ValueError, match="Audio data cannot be empty"):
            processor.process(empty_audio)
    
    def test_noise_profile_learning(self, sample_audio, audio_config):
        """Test noise profile learning."""
        processor = VoiceEnhancer(config=audio_config)
        noise_sample = np.random.randn(16000).astype(np.float32) * 0.1
        processor.learn_noise_profile(noise_sample)
        assert processor.noise_profile is not None
        assert isinstance(processor.noise_profile, np.ndarray)
        assert len(processor.noise_profile) > 0
    
    @pytest.mark.parametrize("noise_strength", [0.0, 0.5, 1.0])
    def test_noise_reduction_levels(self, sample_audio, audio_config, noise_strength):
        """Test different noise reduction strengths."""
        processor = VoiceEnhancer(
            config=audio_config,
            noise_reduction_strength=noise_strength
        )
        
        clean_signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000))
        noise = np.random.randn(16000) * 0.1
        noisy_signal = (clean_signal + noise).astype(np.float32)
        
        processor.learn_noise_profile(noise.astype(np.float32))
        processed = processor.process(noisy_signal)
        
        assert processed.shape == noisy_signal.shape
        assert processed.dtype == np.float32
        
        if noise_strength > 0:
            noise_level_before = np.std(noisy_signal - clean_signal)
            noise_level_after = np.std(processed - clean_signal)
            assert noise_level_after <= noise_level_before
    
    def test_context_manager(self, audio_config):
        """Test processor as context manager."""
        with VoiceEnhancer(config=audio_config) as processor:
            assert processor._is_initialized
            audio = np.random.randn(16000).astype(np.float32)
            result = processor.process(audio)
            assert result is not None
        
        assert not processor._is_initialized
        assert len(processor._cache) == 0


class TestVoiceModel:
    """Test suite for voice synthesis model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        config = ModelConfig(
            hidden_dim=256,
            num_layers=4,
            num_heads=8,
            max_length=64,
            vocab_size=512
        )
        model = VoiceSynthesisModel(config)
        
        assert isinstance(model.token_embedding, nn.Embedding)
        assert model.token_embedding.num_embeddings == 512
        assert model.token_embedding.embedding_dim == 256
        
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
    
    def test_forward_pass(self, voice_model):
        """Test model forward pass."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 256, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        
        logits = voice_model(input_ids, attention_mask)
        assert logits.shape == (batch_size, seq_len, 256)
        assert logits.dtype == torch.float32
    
    def test_generation(self, voice_model):
        """Test autoregressive generation."""
        prompt = torch.tensor([[1, 2, 3]])
        generated = voice_model.generate(
            prompt=prompt,
            max_length=20,
            temperature=0.8,
            top_k=50
        )
        
        assert generated.shape[0] == 1
        assert prompt.shape[1] <= generated.shape[1] <= 20
        assert torch.all(generated[:, :3] == prompt)
    
    def test_checkpoint_save_load(self, voice_model, tmp_path):
        """Test checkpoint saving and loading."""
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        voice_model.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()
        
        loaded_model = VoiceSynthesisModel.from_checkpoint(checkpoint_path)
        assert loaded_model.config == voice_model.config
        
        for (n1, p1), (n2, p2) in zip(
            voice_model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)
    
    @pytest.mark.parametrize("top_k,top_p", [(0, 1.0), (50, 1.0), (0, 0.95), (50, 0.95)])
    def test_sampling_strategies(self, voice_model, top_k, top_p):
        """Test different sampling strategies."""
        logits = torch.randn(1, 256)
        filtered = voice_model._top_k_top_p_filtering(
            logits,
            top_k=top_k,
            top_p=top_p
        )
        
        assert filtered.shape == logits.shape
        if top_k > 0:
            num_valid = (filtered != -float("inf")).sum().item()
            assert num_valid <= top_k
        
        if top_p < 1.0:
            probs = torch.softmax(filtered, dim=-1)
            sorted_probs, _ = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            cutoff_idx = (cumsum > top_p).nonzero(as_tuple=False)
            if cutoff_idx.numel() > 0:
                cutoff = cutoff_idx[0, 0].item()
                assert (filtered != -float("inf")).sum() <= cutoff + 1


@pytest.mark.asyncio
class TestAsyncProcessing:
    """Test suite for async processing."""
    
    async def test_async_file_processing(self, sample_audio, tmp_path):
        """Test async file processing."""
        audio_file = tmp_path / "test.wav"
        np.save(audio_file, sample_audio)
        
        async with AsyncVoiceProcessor() as processor:
            result = await processor.process_voice_file(audio_file)
            assert result["status"] == "success"
            assert "features" in result
            assert result["features"]["duration"] > 0
    
    async def test_batch_processing(self, sample_audio, tmp_path):
        """Test batch processing with mixed success/failure."""
        files = []
        for i in range(3):
            audio_file = tmp_path / f"test_{i}.wav"
            np.save(audio_file, sample_audio)
            files.append(audio_file)
        
        files.append(tmp_path / "missing.wav")
        
        async with AsyncVoiceProcessor() as processor:
            results = await processor.process_voice_batch(files)
            assert len(results) == 4
            for i in range(3):
                assert results[i]["status"] == "success"
            assert results[3]["status"] == "error"
    
    @patch("aiohttp.ClientSession.post")
    async def test_feature_upload(self, mock_post):
        """Test async feature upload."""
        mock_response = MagicMock()
        mock_response.json = MagicMock(return_value={"id": "123", "status": "ok"})
        mock_post.return_value.__aenter__.return_value = mock_response
        
        async with AsyncVoiceProcessor() as processor:
            features = {"duration": 1.5, "energy": 0.8}
            result = await processor._upload_features(features)
            assert result["id"] == "123"
            assert result["status"] == "ok"
            mock_post.assert_called_once()


# ❌ Bad - Poor test structure

def test_processor():
    """Test processor."""
    processor = VoiceEnhancer()
    audio = [1, 2, 3]  # Wrong type
    result = processor.process(audio)
    assert result  # Weak assertion

def test_model():
    """Test model."""
    model = VoiceSynthesisModel()
    # No actual assertions or validations; meaningless test
    print("Model created")
```

* Use **pytest fixtures** for reusable test data.
* Replace broad `assert` statements with **specific checks** (e.g., `pytest.raises`, `assert shape == ...`).
* **Mock** external dependencies (e.g., network calls) in tests to isolate logic.

---

## ⚡ Performance

### Optimization Techniques

```python
# ✅ Good - Performance-optimized code

import numba
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# JIT compilation for numerical operations

@numba.jit(nopython=True, parallel=True)
def compute_spectral_features(
    audio: np.ndarray,
    frame_size: int = 2048,
    hop_size: int = 512
) -> np.ndarray:
    """Compute spectral features using JIT compilation."""
    n_frames = (len(audio) - frame_size) // hop_size + 1
    features = np.zeros((n_frames, frame_size // 2 + 1))
    
    for i in numba.prange(n_frames):
        start = i * hop_size
        frame = audio[start:start + frame_size]
        window = np.hanning(frame_size)
        windowed = frame * window
        spectrum = np.fft.rfft(windowed)
        features[i] = np.abs(spectrum)
    
    return features


# Caching for expensive computations

class CachedVoiceAnalyzer:
    """Voice analyzer with caching."""
    
    def __init__(self, cache_size: int = 128):
        self.cache_size = cache_size
        self._feature_cache = {}
    
    @lru_cache(maxsize=128)
    def _compute_mel_filters(
        self,
        n_filters: int,
        n_fft: int,
        sample_rate: int
    ) -> np.ndarray:
        """Compute mel filterbank (cached)."""
        return librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_filters
        )
    
    def extract_features(
        self,
        audio_path: Path,
        use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract features with optional caching."""
        cache_key = str(audio_path)
        
        if use_cache and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        audio, sr = librosa.load(audio_path, sr=16000)
        with ProcessPoolExecutor(max_workers=4) as executor:
            mfcc_future = executor.submit(self._extract_mfcc, audio, sr)
            pitch_future = executor.submit(self._extract_pitch, audio, sr)
            energy_future = executor.submit(self._extract_energy, audio)
            
            features = {
                "mfcc": mfcc_future.result(),
                "pitch": pitch_future.result(),
                "energy": energy_future.result()
            }
        
        if use_cache and len(self._feature_cache) < self.cache_size:
            self._feature_cache[cache_key] = features
        
        return features
    
    @staticmethod
    def _extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract MFCC features."""
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    @staticmethod
    def _extract_pitch(audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch features."""
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        return pitches.mean(axis=0)
    
    @staticmethod
    def _extract_energy(audio: np.ndarray) -> np.ndarray:
        """Extract energy features."""
        return librosa.feature.rms(y=audio)[0]


class VectorizedProcessor:
    """Processor using vectorized operations."""
    
    @staticmethod
    def normalize_batch(audio_batch: np.ndarray) -> np.ndarray:
        """Normalize a batch of audio samples efficiently."""
        max_vals = np.abs(audio_batch).max(axis=1, keepdims=True)
        max_vals = np.maximum(max_vals, 1e-8)
        return audio_batch / max_vals * 0.95
    
    @staticmethod
    def apply_filters_vectorized(
        audio_batch: np.ndarray,
        filter_coeffs: np.ndarray
    ) -> np.ndarray:
        """Apply filters to audio batch using convolution."""
        n = audio_batch.shape[1] + len(filter_coeffs) - 1
        audio_fft = np.fft.rfft(audio_batch, n=n, axis=1)
        filter_fft = np.fft.rfft(filter_coeffs, n=n)
        result_fft = audio_fft * filter_fft[np.newaxis, :]
        result = np.fft.irfft(result_fft, n=n, axis=1)
        return result[:, :audio_batch.shape[1]]


# Memory-efficient streaming

class StreamingVoiceProcessor:
    """Process large audio files in streaming fashion."""
    
    def __init__(self, chunk_size: int = 16000):
        self.chunk_size = chunk_size
        self.overlap = chunk_size // 4  # 25% overlap
    
    def process_file_streaming(
        self,
        file_path: Path,
        callback: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        """Process large file in chunks.
        
        Args:
            file_path: Path to audio file
            callback: Function to process each chunk
        """
        with soundfile.SoundFile(file_path) as audio_file:
            total_frames = len(audio_file)
            position = 0
            
            while position < total_frames:
                audio_file.seek(max(0, position - self.overlap))
                chunk = audio_file.read(self.chunk_size)
                processed = callback(chunk)
                
                if position > 0:
                    processed = self._blend_overlap(processed, self.overlap)
                
                position += self.chunk_size - self.overlap
    
    @staticmethod
    def _blend_overlap(chunk: np.ndarray, overlap_size: int) -> np.ndarray:
        """Blend overlapping regions using crossfade."""
        fade_in = np.linspace(0, 1, overlap_size)
        fade_out = np.linspace(1, 0, overlap_size)
        
        chunk[:overlap_size] *= fade_in
        
        return chunk
```

* Use **Numba** or **Cython** for JIT-compiled, performance-critical loops.
* **Cache** expensive computations with `functools.lru_cache`.
* Offload independent tasks to **`ProcessPoolExecutor`** for parallelism.

---

## 🔒 Security

### Secure Coding Practices

```python
# ✅ Good - Security-conscious code

import secrets
import hashlib
from cryptography.fernet import Fernet
from pathlib import Path
import tempfile
import os


class SecureFileHandler:
    """Secure file handling for audio uploads."""
    
    def __init__(self, upload_dir: Path, max_file_size: int = 100 * 1024 * 1024):
        """Initialize secure file handler.
        
        Args:
            upload_dir: Directory for uploads
            max_file_size: Maximum allowed file size in bytes
        """
        self.upload_dir = upload_dir
        self.max_file_size = max_file_size
        self.allowed_extensions = {".mp3", ".wav", ".webm", ".ogg"}
        
        # Ensure upload directory is secure
        self.upload_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    
    def validate_file(self, file_path: Path, user_id: str) -> None:
        """Validate uploaded file for security.
        
        Args:
            file_path: Path to uploaded file
            user_id: ID of user uploading file
            
        Raises:
            SecurityError: If file fails validation
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise SecurityError(
                f"File too large: {file_size} bytes (max: {self.max_file_size})"
            )
        
        if file_path.suffix.lower() not in self.allowed_extensions:
            raise SecurityError(f"Invalid file extension: {file_path.suffix}")
        
        mime_type = self._get_mime_type(file_path)
        if not mime_type.startswith("audio/"):
            raise SecurityError(f"Invalid MIME type: {mime_type}")
        
        if ".." in str(file_path) or file_path.is_absolute():
            raise SecurityError("Path traversal detected")
        
        if not self._verify_ownership(file_path, user_id):
            raise SecurityError("Unauthorized file access")
    
    def secure_save(
        self,
        file_data: bytes,
        original_name: str,
        user_id: str
    ) -> Path:
        """Securely save uploaded file.
        
        Args:
            file_data: File content
            original_name: Original filename
            user_id: User ID
            
        Returns:
            Path to saved file
        """
        file_id = secrets.token_urlsafe(32)
        extension = Path(original_name).suffix.lower()
        
        if extension not in self.allowed_extensions:
            raise SecurityError(f"Invalid extension: {extension}")
        
        user_dir = self.upload_dir / self._hash_user_id(user_id)
        user_dir.mkdir(mode=0o700, exist_ok=True)
        
        file_path = user_dir / f"{file_id}{extension}"
        
        with tempfile.NamedTemporaryFile(
            dir=user_dir,
            delete=False,
            mode="wb"
        ) as tmp_file:
            tmp_file.write(file_data)
            tmp_path = Path(tmp_file.name)
        
        tmp_path.chmod(0o600)
        tmp_path.rename(file_path)
        
        logger.info(
            f"File saved: user={user_id}, "
            f"file={file_path.name}, "
            f"size={len(file_data)}"
        )
        
        return file_path
    
    @staticmethod
    def _get_mime_type(file_path: Path) -> str:
        """Get MIME type using python-magic."""
        import magic
        mime = magic.Magic(mime=True)
        return mime.from_file(str(file_path))
    
    @staticmethod
    def _hash_user_id(user_id: str) -> str:
        """Hash user ID for directory name."""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]
    
    def _verify_ownership(self, file_path: Path, user_id: str) -> bool:
        """Verify user owns the file."""
        expected_dir = self.upload_dir / self._hash_user_id(user_id)
        return file_path.parent == expected_dir


class SecureDataProcessor:
    """Process sensitive data securely."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """Initialize secure processor.
        
        Args:
            encryption_key: Encryption key (generated if not provided)
        """
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        
        self.cipher = Fernet(encryption_key)
    
    def process_sensitive_audio(
        self,
        audio_data: np.ndarray,
        metadata: Dict[str, Any]
    ) -> bytes:
        """Process and encrypt sensitive audio data.
        
        Args:
            audio_data: Audio samples
            metadata: Audio metadata
            
        Returns:
            Encrypted data package
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("Audio data must be numpy array")
        
        safe_metadata = self._sanitize_metadata(metadata)
        package = {
            "audio": audio_data.tobytes(),
            "dtype": str(audio_data.dtype),
            "shape": audio_data.shape,
            "metadata": safe_metadata
        }
        
        serialized = pickle.dumps(package)
        encrypted = self.cipher.encrypt(serialized)
        
        return encrypted
    
    def decrypt_audio_package(self, encrypted_data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Decrypt and unpack audio data.
        
        Args:
            encrypted_data: Encrypted data package
            
        Returns:
            Tuple of (audio_data, metadata)
        """
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            package = pickle.loads(decrypted)
            
            audio_bytes = package["audio"]
            dtype = np.dtype(package["dtype"])
            shape = package["shape"]
            
            audio_data = np.frombuffer(audio_bytes, dtype=dtype).reshape(shape)
            return audio_data, package["metadata"]
            
        except Exception as e:
            logger.error("Decryption failed", exc_info=True)
            raise SecurityError("Failed to decrypt audio package") from e
    
    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive information from metadata."""
        sensitive_keys = {
            "user_id", "email", "ip_address", "api_key",
            "password", "token", "secret"
        }
        sanitized = {}
        for key, value in metadata.items():
            if key.lower() not in sensitive_keys:
                if isinstance(value, dict):
                    sanitized[key] = SecureDataProcessor._sanitize_metadata(value)
                else:
                    sanitized[key] = value
        return sanitized
```

* **Enforce** file size and extension checks.
* **Prevent** path traversal.
* **Encrypt** sensitive data with a **strong cipher**.
* Remove sensitive fields (e.g., `user_id`, `api_key`) from metadata.

---

# End of Style Guide


