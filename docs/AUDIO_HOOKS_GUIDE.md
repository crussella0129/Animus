# Audio Hooks Guide for Animus

This guide explains how to add audio feedback to Animus — playing .wav files during processing and fanfares when events complete. This is designed for when you're learning to code and want to implement these features yourself.

## Overview

Audio hooks let you play sounds at key moments:
- **Processing sounds** — ambient loops or click sounds while the agent is thinking
- **Event fanfares** — celebration sounds when tasks complete, tests pass, commits succeed
- **Error sounds** — alert sounds when something goes wrong

## Where to Add Audio Code

### Key Files to Understand

1. **`src/core/agent.py`** — Main agent loop where LLM calls happen
2. **`src/core/session.py`** — Session management, streaming responses
3. **`src/tools/base.py`** — Tool execution (where you can hook into tool calls)
4. **`src/main.py`** — CLI commands (where you can add fanfares for command completion)

## Python Basics You Need

### Playing Audio in Python

The simplest way to play audio in Python is using the `playsound` library:

```python
# Install: pip install playsound
from playsound import playsound

# Play a sound (blocks until finished)
playsound("sounds/thinking.wav")

# Play in background (non-blocking)
from threading import Thread
Thread(target=playsound, args=("sounds/thinking.wav",), daemon=True).start()
```

For more control, use `pygame.mixer`:

```python
# Install: pip install pygame
import pygame

# Initialize mixer
pygame.mixer.init()

# Load and play a sound
sound = pygame.mixer.Sound("sounds/fanfare.wav")
sound.play()

# Play music (for loops)
pygame.mixer.music.load("sounds/background.wav")
pygame.mixer.music.play(loops=-1)  # -1 = infinite loop
pygame.mixer.music.stop()
```

## Implementation Examples

### Example 1: Play Sound While Agent is Thinking

Add to `src/core/agent.py` in the `Agent.run()` method:

```python
def run(self, user_message: str) -> str:
    """Run the agent on a user message."""

    # Start thinking sound
    thinking_sound = None
    if config.audio.thinking_enabled:  # You'd add this config
        import pygame
        pygame.mixer.init()
        thinking_sound = pygame.mixer.Sound("sounds/thinking.wav")
        thinking_sound.play(loops=-1)  # Loop until stopped

    try:
        # ... existing agent logic ...
        response = self._call_llm(messages)

        return response
    finally:
        # Stop thinking sound
        if thinking_sound:
            thinking_sound.stop()
```

### Example 2: Fanfare When Tool Execution Succeeds

Add to `src/tools/base.py` in the `Tool.execute()` wrapper:

```python
def execute(self, params: dict[str, Any]) -> str:
    """Execute the tool with the given parameters."""
    try:
        result = self._execute(params)

        # Play success fanfare for important tools
        if self.name in ["git_commit", "run_tests"]:
            play_fanfare("success")

        return result
    except Exception as e:
        # Play error sound
        play_fanfare("error")
        raise

def play_fanfare(event_type: str):
    """Play a fanfare sound for an event."""
    from threading import Thread
    from playsound import playsound

    sounds = {
        "success": "sounds/success.wav",
        "error": "sounds/error.wav",
        "complete": "sounds/complete.wav",
    }

    if event_type in sounds:
        # Play in background thread so it doesn't block
        Thread(
            target=playsound,
            args=(sounds[event_type],),
            daemon=True
        ).start()
```

### Example 3: Event-Based Audio System

Create a new file `src/audio/events.py`:

```python
"""Audio event system for Animus."""

from enum import Enum
from pathlib import Path
from threading import Thread
import pygame

class AudioEvent(Enum):
    """Audio events that can trigger sounds."""
    AGENT_START = "agent_start"
    AGENT_THINKING = "agent_thinking"
    AGENT_COMPLETE = "agent_complete"
    TOOL_EXECUTE = "tool_execute"
    TOOL_SUCCESS = "tool_success"
    TOOL_ERROR = "tool_error"
    TEST_PASS = "test_pass"
    TEST_FAIL = "test_fail"
    COMMIT_SUCCESS = "commit_success"
    BUILD_SUCCESS = "build_success"

class AudioManager:
    """Manages audio playback for Animus events."""

    def __init__(self, sounds_dir: Path):
        """Initialize the audio manager.

        Args:
            sounds_dir: Directory containing .wav files
        """
        self.sounds_dir = sounds_dir
        self.enabled = True
        self._current_loop = None

        # Initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

        # Map events to sound files
        self.sound_map = {
            AudioEvent.AGENT_START: "agent_start.wav",
            AudioEvent.AGENT_THINKING: "thinking_loop.wav",
            AudioEvent.AGENT_COMPLETE: "agent_complete.wav",
            AudioEvent.TOOL_SUCCESS: "tool_success.wav",
            AudioEvent.TOOL_ERROR: "error.wav",
            AudioEvent.TEST_PASS: "test_pass.wav",
            AudioEvent.TEST_FAIL: "test_fail.wav",
            AudioEvent.COMMIT_SUCCESS: "commit_success.wav",
            AudioEvent.BUILD_SUCCESS: "fanfare.wav",
        }

    def play(self, event: AudioEvent, loop: bool = False):
        """Play a sound for an event.

        Args:
            event: The event that occurred
            loop: Whether to loop the sound (for ambient/thinking sounds)
        """
        if not self.enabled:
            return

        sound_file = self.sound_map.get(event)
        if not sound_file:
            return

        sound_path = self.sounds_dir / sound_file
        if not sound_path.exists():
            return

        if loop:
            # Play as background music (looping)
            pygame.mixer.music.load(str(sound_path))
            pygame.mixer.music.play(loops=-1)
            self._current_loop = event
        else:
            # Play as one-shot sound effect
            sound = pygame.mixer.Sound(str(sound_path))
            sound.play()

    def stop_loop(self):
        """Stop any currently looping sound."""
        if self._current_loop:
            pygame.mixer.music.stop()
            self._current_loop = None

    def set_volume(self, volume: float):
        """Set the volume (0.0 to 1.0)."""
        pygame.mixer.music.set_volume(volume)


# Global audio manager instance
_audio_manager: AudioManager | None = None

def init_audio(sounds_dir: Path) -> AudioManager:
    """Initialize the global audio manager."""
    global _audio_manager
    _audio_manager = AudioManager(sounds_dir)
    return _audio_manager

def play_event(event: AudioEvent, loop: bool = False):
    """Play an audio event (convenience function)."""
    if _audio_manager:
        _audio_manager.play(event, loop=loop)

def stop_loop():
    """Stop any looping audio."""
    if _audio_manager:
        _audio_manager.stop_loop()
```

### Example 4: Using the Audio System in Agent

Modify `src/core/agent.py`:

```python
from src.audio.events import AudioEvent, play_event, stop_loop

class Agent:
    def run(self, user_message: str) -> str:
        """Run the agent on a user message."""

        # Play start sound
        play_event(AudioEvent.AGENT_START)

        # Start thinking loop
        play_event(AudioEvent.AGENT_THINKING, loop=True)

        try:
            # ... agent logic ...
            response = self._call_llm(messages)

            # Stop thinking loop
            stop_loop()

            # Play completion sound
            play_event(AudioEvent.AGENT_COMPLETE)

            return response
        except Exception as e:
            stop_loop()
            play_event(AudioEvent.TOOL_ERROR)
            raise
```

### Example 5: Adding to Configuration

Add to `src/config.py`:

```python
class AudioConfig(BaseModel):
    """Audio configuration."""
    enabled: bool = True
    sounds_dir: Path = Path.home() / ".animus" / "sounds"
    volume: float = 0.5
    thinking_loop: bool = True
    fanfares: bool = True

class AnimusConfig(BaseModel):
    # ... existing fields ...
    audio: AudioConfig = Field(default_factory=AudioConfig)
```

## Sound File Organization

Create a `sounds/` directory in your Animus config folder:

```
~/.animus/sounds/
├── agent_start.wav        # Quick blip when agent starts
├── thinking_loop.wav      # Ambient loop during processing
├── agent_complete.wav     # Satisfying "done" sound
├── tool_success.wav       # Tool executed successfully
├── error.wav              # Something went wrong
├── test_pass.wav          # Tests passed
├── test_fail.wav          # Tests failed
├── commit_success.wav     # Git commit successful
└── fanfare.wav            # Major accomplishment
```

## Finding/Creating Sound Effects

- **Free sounds**: [freesound.org](https://freesound.org)
- **Create your own**: Audacity (free audio editor)
- **AI generation**: Use Animus to generate sounds with AI tools
- **Keep them short**: 0.5-2 seconds for events, <10 seconds for fanfares

## Tips for Implementation

1. **Start simple** — Just add one sound to one event first
2. **Use threading** — Audio should never block the agent
3. **Make it configurable** — Users should be able to disable sounds
4. **Keep volumes low** — Default to 30-50% volume
5. **Handle missing files gracefully** — Don't crash if a sound file is missing
6. **Test with headphones** — Make sure sounds aren't annoying
7. **Consider accessibility** — Some users may want visual indicators instead

## Code You Need to Learn

To implement audio hooks, focus on learning:

1. **Python basics** — Functions, classes, imports
2. **File I/O** — Reading files, checking if files exist (`Path.exists()`)
3. **Threading** — Running code in background (`threading.Thread`)
4. **Configuration** — Reading config files (Animus uses `pydantic`)
5. **Error handling** — Try/except blocks to handle missing sounds gracefully

## Testing Your Audio

Simple test script:

```python
# test_audio.py
from pathlib import Path
from src.audio.events import init_audio, play_event, AudioEvent
import time

# Initialize
sounds_dir = Path.home() / ".animus" / "sounds"
init_audio(sounds_dir)

# Test each sound
events = [
    AudioEvent.AGENT_START,
    AudioEvent.TOOL_SUCCESS,
    AudioEvent.COMMIT_SUCCESS,
]

for event in events:
    print(f"Playing: {event.value}")
    play_event(event)
    time.sleep(2)

# Test looping
print("Playing thinking loop (5 seconds)...")
play_event(AudioEvent.AGENT_THINKING, loop=True)
time.sleep(5)
stop_loop()

print("Done!")
```

## Next Steps

1. **Create the sounds directory**: `mkdir -p ~/.animus/sounds`
2. **Add a few .wav files** to test with
3. **Implement the AudioManager class** from Example 3
4. **Add audio hooks** to one part of the codebase (start with agent.py)
5. **Test and refine** — adjust volumes, timing, and which events have sounds
6. **Make it configurable** — add settings to disable/enable specific sounds

## Questions to Explore

- Should sounds play during background agent tasks?
- Should there be different sounds for different tool categories?
- Should sounds be quieter/disabled in automated environments (CI/CD)?
- Should there be a "mute" command or keyboard shortcut?

---

**Remember**: Audio feedback should enhance the experience, not distract from it. Start minimal and add more as you find what works for you!
