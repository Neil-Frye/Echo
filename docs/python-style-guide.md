# Python Style Guide

## 📋 Table of Contents

- [Introduction](#introduction)
- [General Principles](#general-principles)
- [Code Layout](#code-layout)
- [Naming Conventions](#naming-conventions)
- [Type Hints](#type-hints)
- [Functions and Methods](#functions-and-methods)
- [Classes](#classes)
- [Error Handling](#error-handling)
- [Async Programming](#async-programming)
- [AI/ML Guidelines](#aiml-guidelines)
- [Testing](#testing)
- [Performance](#performance)
- [Security](#security)

## 🎯 Introduction

This style guide defines Python coding standards for EthernalEcho's AI and backend services. We follow PEP 8 as our foundation, with additional guidelines specific to our domain.

## 🏗️ General Principles

1. **Clarity over Cleverness**: Write code that's easy to understand
2. **Type Safety**: Use type hints everywhere
3. **Explicit over Implicit**: Follow the Zen of Python
4. **Documentation**: Every public API must be documented
5. **Testability**: Design code that's easy to test

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
recording = False  # Ambiguous boolean name
MAX_RETRY = 3  # Should be lowercase
audioSampleRate = 16000  # Should be snake_case

def calc(s: List[Any]) -> float:  # Unclear function name
    return sum(s.q for s in s) / len(s)
```

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
sample_rate = 16000  # Should be uppercase
MAXDURATION = 300  # Should be snake_case
supported_formats = ["mp3", "wav"]  # Should be uppercase and tuple
```

## 🔧 Type Hints

### Basic Type Hints

```python
# ✅ Good - Comprehensive type hints
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from datetime import datetime
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
        # Process each sample
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

### Advanced Type Hints

```python
# ✅ Good - Advanced type usage
from typing import TypeVar, Generic, Protocol, Literal, TypedDict, Callable
from typing_extensions import ParamSpec
import numpy.typing as npt

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

### Type Aliases

```python
# ✅ Good - Meaningful type aliases
from typing import Dict, List, Tuple, Union
import numpy as np
import torch

# Type aliases for clarity
AudioArray = npt.NDArray[np.float32]
AudioTensor = torch.Tensor
SampleRate = int
Duration = float
EmbeddingVector = npt.NDArray[np.float32]
FeatureDict = Dict[str, Union[float, List[float]]]

def extract_audio_features(
    audio: AudioArray,
    sample_rate: SampleRate
) -> Tuple[EmbeddingVector, FeatureDict]:
    """Extract audio features and embeddings."""
    # Implementation
    pass
```

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
    
    # Create model with configuration
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

## 🏛️ Classes

### Class Design

```python
# ✅ Good - Well-designed class
from __future__ import annotations
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
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
        """Set up audio filters."""
        # Implementation details
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
            noise_reduction_strength: Strength of noise reduction (0-1)
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
        
        # Spectral subtraction implementation
        spectrum = self._compute_spectrum(audio)
        clean_spectrum = spectrum - self.noise_reduction_strength * self.noise_profile
        clean_spectrum = np.maximum(clean_spectrum, 0)
        
        return self._inverse_spectrum(clean_spectrum)
    
    def _enhance_voice(self, audio: np.ndarray) -> np.ndarray:
        """Enhance voice frequencies."""
        # Implementation details
        return audio
    
    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove silence from audio."""
        # Implementation details
        return audio
    
    def _compute_spectrum(self, audio: np.ndarray) -> np.ndarray:
        """Compute frequency spectrum."""
        # Implementation details
        return np.fft.rfft(audio)
    
    def _inverse_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute inverse spectrum."""
        # Implementation details
        return np.fft.irfft(spectrum)


# ❌ Bad - Poor class design
class voice_enhancer:  # Should be PascalCase
    def __init__(self):
        self.config = {}  # No type hints
        self.initialized = False
    
    def process(self, audio):  # No type hints
        # No validation
        return audio
    
    # No separation of concerns
    def do_everything(self, audio, config, noise_sample):
        # Too many responsibilities
        pass
```

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
    except:  # Bare except
        print("Error occurred")  # Just printing
        return None  # Returns None on error
```

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
        
        # Handle partial failures
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
        # Convert bytes to numpy array
        audio = np.frombuffer(audio_data, dtype=np.int16)
        
        # Extract features
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_hidden_states: Whether to return hidden states
            
        Returns:
            Output logits or tuple of (logits, hidden_states)
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device)
        position_embeds = self.position_embedding(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = F.dropout(hidden_states, p=self.config.dropout, training=self.training)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to transformer format (True = masked)
            attention_mask = ~attention_mask.bool()
        
        # Pass through transformer
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=attention_mask
        )
        
        # Output projection
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
            Generated token sequences
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Initialize with prompt or start token
        if prompt is not None:
            generated = prompt.to(device)
        else:
            generated = torch.zeros((1, 1), dtype=torch.long, device=device)
        
        for _ in range(max_length - generated.shape[1]):
            # Get model predictions
            logits = self(generated)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k and top-p filtering
            filtered_logits = self._top_k_top_p_filtering(
                next_token_logits,
                top_k=top_k,
                top_p=top_p
            )
            
            # Sample next token
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
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
        """Filter logits using top-k and/or top-p filtering.
        
        Args:
            logits: Logits distribution shape [batch_size, vocab_size]
            top_k: Keep only top k tokens with highest probability
            top_p: Keep top tokens with cumulative probability >= top_p
            filter_value: Value to assign to filtered tokens
            
        Returns:
            Filtered logits
        """
        if top_k > 0:
            # Remove all tokens with a probability less than the top_k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value
        
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted tensors to original indexing
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
        
        # Initialize wandb
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
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.gradient_clip
            )
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/gradient_norm": self._get_gradient_norm()
                }, step=self.global_step)
        
        return {
            "loss": total_loss / total_tokens,
            "perplexity": torch.exp(torch.tensor(total_loss / total_tokens)).item()
        }
    
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
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            
            # Update metrics
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()
        
        val_loss = total_loss / total_tokens
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({
                "val/loss": val_loss,
                "val/perplexity": torch.exp(torch.tensor(val_loss)).item()
            }, step=self.global_step)
        
        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            if self.checkpoint_dir:
                self.model.save_checkpoint(
                    self.checkpoint_dir / "best_model.pth"
                )
        
        return {
            "loss": val_loss,
            "perplexity": torch.exp(torch.tensor(val_loss)).item()
        }
    
    def train(self, num_epochs: int) -> None:
        """Train model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        if self.use_wandb:
            wandb.watch(self.model)
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            logger.info(f"Train metrics: {train_metrics}")
            
            # Validate epoch
            val_metrics = self.validate()
            if val_metrics:
                logger.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            if self.checkpoint_dir:
                self.model.save_checkpoint(
                    self.checkpoint_dir / f"epoch_{epoch + 1}.pth"
                )
        
        if self.use_wandb:
            wandb.finish()
    
    def _get_gradient_norm(self) -> float:
        """Calculate total gradient norm."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
        return total_norm


# ❌ Bad - Poor training loop
def train_model(model, data, epochs):
    """Train model."""
    for epoch in range(epochs):
        for batch in data:
            loss = model(batch)
            loss.backward()
            # Missing optimizer step, scheduler, logging, validation, etc.
```

## 🧪 Testing

### Test Structure

```python
# ✅ Good - Well-structured tests
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from services.ai.processors.audio_processor import AudioProcessor
from services.ai.models.voice_synthesis import VoiceSynthesisModel, ModelConfig

# Fixtures
@pytest.fixture
def sample_audio_data() -> np.ndarray:
    """Fixture for sample audio data."""
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 1.0  # seconds
    frequency = 440  # Hz
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2. * np.pi * frequency * t)
    return audio.astype(np.float32)


@pytest.fixture
def mock_model_config() -> ModelConfig:
    """Fixture for a mock model configuration."""
    return ModelConfig(
        hidden_dim=16,
        num_layers=2,
        num_heads=2,
        max_length=100,
        vocab_size=256
    )


@pytest.fixture
def mock_voice_model(mock_model_config: ModelConfig) -> VoiceSynthesisModel:
    """Fixture for a mock voice synthesis model."""
    model = VoiceSynthesisModel(mock_model_config)
    # Initialize with dummy weights if needed
    return model


# Unit tests for AudioProcessor
class TestAudioProcessor:
    def test_process_audio_sample_valid(self, sample_audio_data: np.ndarray):
        """Test processing a valid audio sample."""
        processor = AudioProcessor(sample_rate=16000)
        processed_audio = processor.process(sample_audio_data)
        
        assert isinstance(processed_audio, np.ndarray)
        assert processed_audio.ndim == 1
        assert len(processed_audio) > 0
        # Add more specific assertions based on processing logic
    
    def test_process_audio_sample_empty(self):
        """Test processing an empty audio sample."""
        processor = AudioProcessor(sample_rate=16000)
        with pytest.raises(ValueError, match="Audio data cannot be empty"):
            processor.process(np.array([], dtype=np.float32))
    
    @pytest.mark.parametrize("invalid_rate", [0, -100, 10.5])
    def test_audio_processor_invalid_sample_rate(self, invalid_rate: Any):
        """Test AudioProcessor with invalid sample rates."""
        with pytest.raises(ValueError, match="Sample rate must be positive integer"):
            AudioProcessor(sample_rate=invalid_rate)


# Unit tests for VoiceSynthesisModel
class TestVoiceSynthesisModel:
    def test_model_initialization(self, mock_model_config: ModelConfig):
        """Test model initialization."""
        model = VoiceSynthesisModel(mock_model_config)
        assert isinstance(model, nn.Module)
        assert model.config == mock_model_config
        assert model.token_embedding.embedding_dim == mock_model_config.hidden_dim
        assert model.transformer.num_layers == mock_model_config.num_layers
    
    @patch("torch.save")
    def test_save_checkpoint(
        self,
        mock_torch_save: MagicMock,
        mock_voice_model: VoiceSynthesisModel
    ):
        """Test saving a model checkpoint."""
        checkpoint_path = Path("/tmp/test_checkpoint.pth")
        mock_voice_model.save_checkpoint(checkpoint_path)
        
        mock_torch_save.assert_called_once()
        call_args, call_kwargs = mock_torch_save.call_args
        
        saved_checkpoint = call_args[0]
        saved_path = call_args[1]
        
        assert "model_state_dict" in saved_checkpoint
        assert "config" in saved_checkpoint
        assert saved_checkpoint["config"] == mock_voice_model.config.to_dict()
        assert saved_path == checkpoint_path
    
    @patch("torch.load")
    @patch("services.ai.models.voice_synthesis.VoiceSynthesisModel")
    def test_from_checkpoint(
        self,
        MockVoiceSynthesisModel: MagicMock,
        mock_torch_load: MagicMock,
        mock_model_config: ModelConfig
    ):
        """Test loading a model from a checkpoint."""
        checkpoint_path = Path("/tmp/test_checkpoint.pth")
        mock_checkpoint_data = {
            "model_state_dict": {"dummy_key": "dummy_value"},
            "config": mock_model_config.to_dict()
        }
        mock_torch_load.return_value = mock_checkpoint_data
        
        loaded_model = VoiceSynthesisModel.from_checkpoint(checkpoint_path)
        
        mock_torch_load.assert_called_once_with(checkpoint_path, map_location="cpu")
        MockVoiceSynthesisModel.assert_called_once_with(mock_model_config)
        loaded_model.load_state_dict.assert_called_once_with(
            mock_checkpoint_data["model_state_dict"]
        )
        assert loaded_model == MockVoiceSynthesisModel.return_value


# Integration tests (example)
@pytest.mark.integration
def test_audio_processing_pipeline(sample_audio_data: np.ndarray):
    """Test the full audio processing pipeline."""
    processor = AudioProcessor(sample_rate=16000)
    model_config = ModelConfig(hidden_dim=16, num_layers=2, num_heads=2)
    model = VoiceSynthesisModel(model_config)
    
    # Simulate processing and encoding
    processed_audio = processor.process(sample_audio_data)
    audio_tensor = torch.from_numpy(processed_audio).unsqueeze(0) # Add batch dim
    embedding = model(audio_tensor, return_hidden_states=True)[1] # Get hidden state
    
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (1, audio_tensor.shape[1], model_config.hidden_dim)
    # Add more assertions based on expected output
```

### Test Utilities

```python
# ✅ Good - Reusable test utilities
import numpy as np
import torch
from unittest.mock import MagicMock

def create_mock_audio_array(
    duration_seconds: float,
    sample_rate: int = 16000
) -> np.ndarray:
    """Create a mock audio numpy array."""
    num_samples = int(duration_seconds * sample_rate)
    # Generate random audio data between -1 and 1
    audio = 2 * np.random.rand(num_samples) - 1
    return audio.astype(np.float32)


def create_mock_voice_sample(
    sample_id: str = "test-sample",
    duration: float = 30.0,
    quality: float = 0.9,
    audio_url: str = "https://example.com/audio.wav"
) -> MagicMock:
    """Create a mock VoiceSample object."""
    mock_sample = MagicMock()
    mock_sample.id = sample_id
    mock_sample.duration = duration
    mock_sample.quality = quality
    mock_sample.audio_url = audio_url
    return mock_sample


def create_mock_model_state_dict(
    config: ModelConfig
) -> Dict[str, torch.Tensor]:
    """Create a mock model state dictionary."""
    # This is a simplified example; a real state dict would be more complex
    return {
        "token_embedding.weight": torch.randn(
            config.vocab_size,
            config.hidden_dim
        ),
        "position_embedding.weight": torch.randn(
            config.max_length,
            config.hidden_dim
        ),
        # ... other layers
    }
```

## ⚡ Performance

### Profiling

```python
# ✅ Good - Use profiling tools
import cProfile
import pstats
import io
import logging

logger = logging.getLogger(__name__)

def profile_function(func: Callable, *args: Any, **kwargs: Any) -> Any:
    """Profile a function and print results."""
    pr = cProfile.Profile()
    pr.enable()
    
    result = func(*args, **kwargs)
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()
    
    logger.info(f"Profiling results for {func.__name__}:\n{s.getvalue()}")
    
    return result

# Usage
# profile_function(process_large_audio_file, "path/to/file.wav")
```

### Optimization Techniques

```python
# ✅ Good - Optimized code
import numpy as np
import torch

def batch_process_audio(
    audio_list: List[np.ndarray],
    sample_rate: int = 16000
) -> torch.Tensor:
    """Process a batch of audio samples efficiently."""
    # Pad sequences to max length
    max_len = max(len(a) for a in audio_list)
    padded_audio = np.zeros((len(audio_list), max_len), dtype=np.float32)
    attention_masks = torch.zeros((len(audio_list), max_len), dtype=torch.long)
    
    for i, audio in enumerate(audio_list):
        padded_audio[i, :len(audio)] = audio
        attention_masks[i, :len(audio)] = 1 # 1 for real tokens, 0 for padding
    
    # Convert to torch tensor and move to device
    audio_tensor = torch.from_numpy(padded_audio).to("cuda")
    attention_masks = attention_masks.to("cuda")
    
    # Process batch on GPU
    processed_tensor = process_audio_batch_on_gpu(audio_tensor, attention_masks)
    
    return processed_tensor

# ❌ Bad - Inefficient code
def process_audio_list_slow(audio_list: List[np.ndarray]) -> List[torch.Tensor]:
    """Process audio list inefficiently."""
    results = []
    for audio in audio_list:
        # Process each sample individually on CPU
        processed = process_single_audio_on_cpu(audio)
        results.append(torch.from_numpy(processed))
    return results
```

## 🔒 Security

### Input Validation

```python
# ✅ Good - Robust input validation
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import re

class VoiceUploadRequest(BaseModel):
    """Schema for voice upload request."""
    user_id: str = Field(..., description="ID of the user uploading")
    file_name: str = Field(..., description="Original name of the uploaded file")
    file_size: int = Field(..., gt=0, description="Size of the uploaded file in bytes")
    audio_format: str = Field(..., description="Format of the audio file (e.g., 'wav', 'mp3')")
    duration: float = Field(..., gt=0, description="Duration of the audio in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    
    @validator("user_id")
    def validate_user_id(cls, v):
        if not re.match(r"^[a-f0-9]{24}$", v): # Example: MongoDB ObjectId format
            raise ValueError("Invalid user ID format")
        return v
    
    @validator("file_name")
    def validate_file_name(cls, v):
        if not re.match(r"^[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+$", v):
            raise ValueError("Invalid file name format")
        return v
    
    @validator("audio_format")
    def validate_audio_format(cls, v):
        if v.lower() not in SUPPORTED_AUDIO_FORMATS:
            raise ValueError(f"Unsupported audio format: {v}")
        return v
    
    @validator("duration")
    def validate_duration(cls, v):
        if v > MAX_AUDIO_DURATION:
            raise ValueError(f"Audio duration exceeds maximum allowed: {MAX_AUDIO_DURATION}s")
        return v


# Usage in FastAPI endpoint
from fastapi import APIRouter, UploadFile, File, Depends

router = APIRouter()

@router.post("/voice/upload")
async def upload_voice(
    user_id: str,
    file: UploadFile = File(...),
    duration: float = Field(..., gt=0),
    audio_format: str = Field(...)
):
    # Validate input using Pydantic model
    request_data = VoiceUploadRequest(
        user_id=user_id,
        file_name=file.filename,
        file_size=file.size,
        audio_format=audio_format,
        duration=duration
    )
    
    # Process the valid file
    content = await file.read()
    # ... further processing
    
    return {"message": "Upload successful"}
```

### Secure Data Handling

```python
# ✅ Good - Secure data handling
import hashlib
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.fernet import Fernet
import base64
import json
import logging

logger = logging.getLogger(__name__)

class SecureDataHandler:
    """Handles secure encryption and decryption of sensitive data."""
    
    def __init__(self, encryption_key: str):
        """Initialize handler with a base64-encoded encryption key."""
        try:
            self._fernet = Fernet(encryption_key)
        except Exception as e:
            logger.critical(f"Invalid encryption key provided: {e}")
            raise ValueError("Invalid encryption key format") from e
    
    def encrypt(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary of data."""
        try:
            json_data = json.dumps(data)
            encrypted_data = self._fernet.encrypt(json_data.encode("utf-8"))
            return encrypted_data.decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encrypt data: {e}", exc_info=True)
            raise RuntimeError("Data encryption failed") from e
    
    def decrypt(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt a base64-encoded encrypted string."""
        try:
            decrypted_data = self._fernet.decrypt(encrypted_data.encode("utf-8"))
            return json.loads(decrypted_data.decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to decrypt data: {e}", exc_info=True)
            raise RuntimeError("Data decryption failed") from e
    
    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet encryption key."""
        return Fernet.generate_key().decode("utf-8")
    
    @staticmethod
    def derive_key_from_password(password: str, salt: bytes) -> str:
        """Derive an encryption key from a password and salt."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000, # Recommended iterations
            backend=default_backend()
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
        return key.decode("utf-8")


# Usage
# key = SecureDataHandler.generate_key() # Store this securely!
# handler = SecureDataHandler(key)

# sensitive_data = {"credit_card": "...", "ssn": "..."}
# encrypted_string = handler.encrypt(sensitive_data)

# decrypted_data = handler.decrypt(encrypted_string)
```

## 📚 Resources

- [PEP 8 -- Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 257 -- Docstring Conventions](https://peps.python.org/pep-0257/)
- [PEP 484 -- Type Hints](https://peps.python.org/pep-0484/)
- [PEP 526 -- Syntax for Variable Annotations](https://peps.python.org/pep-0526/)
- [PEP 586 -- Literal Types](https://peps.python.org/pep-0586/)
- [PEP 589 -- TypedDict](https://peps.python.org/pep-0589/)
- [PEP 612 -- Parameter Specification Variables](https://peps.python.org/pep-0612/)
- [PEP 655 -- Marking individual `TypedDict` items as required or potentially missing](https://peps.python.org/pep-0655/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Hitchhiker's Guide to Python!](https://docs.python-guide.org/)

---

*Last updated: January 2024*
