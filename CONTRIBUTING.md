# Contributing to EthernalEcho

First off, thank you for considering contributing to EthernalEcho! It's people like you that help us build a platform that truly serves families in their time of need. 

This document provides guidelines for contributing to EthernalEcho. Following these guidelines helps maintain code quality, ensures a positive community experience, and speeds up the review process.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## 📜 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:

- **Be Respectful**: Treat everyone with respect. We're building technology for sensitive situations.
- **Be Empathetic**: Remember our users are often dealing with loss and grief.
- **Be Professional**: Maintain professional conduct in all interactions.
- **Be Inclusive**: Welcome contributors from all backgrounds.
- **Be Patient**: Both with others and with the review process.

Violations should be reported to conduct@ethernalecho.com.

## 🤝 How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**Submit a bug report through GitHub Issues with:**
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, browser, version)
- Screenshots if applicable
- Error messages and logs

### Suggesting Enhancements

**Submit enhancement suggestions through GitHub Issues with:**
- Clear use case explanation
- Detailed description of the solution
- Mockups or examples if applicable
- Why this would benefit users

### Contributing Code

We welcome code contributions for:
- Bug fixes
- New features (discuss first)
- Performance improvements
- Test coverage improvements
- Documentation updates

### Other Contributions

- **Documentation**: Help improve our docs
- **Translation**: Help translate the platform
- **Testing**: Help test new features
- **Design**: Contribute UI/UX improvements

## 🛠️ Development Setup

### Prerequisites

- Node.js 20.x or higher
- Python 3.11 or higher
- Docker Desktop
- Git with GPG signing configured
- IDE with ESLint and Prettier support

### Local Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR-USERNAME/ethernalecho.git
   cd ethernalecho
   ```

2. **Install Dependencies**
   ```bash
   # Install Node dependencies
   npm install
   
   # Install Python dependencies
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Environment Configuration**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your configuration
   ```

4. **Start Development Services**
   ```bash
   # Start Docker services (databases, Redis, etc.)
   docker-compose -f docker-compose.dev.yml up -d
   
   # Run database migrations
   npm run migrate:dev
   
   # Seed development data
   npm run seed:dev
   ```

5. **Start Development Servers**
   ```bash
   # Terminal 1: Start web app
   npm run dev:web
   
   # Terminal 2: Start API services
   npm run dev:api
   
   # Terminal 3: Start AI services
   python services/ai/server.py --dev
   ```

6. **Verify Setup**
   - Web app: http://localhost:3000
   - API: http://localhost:4000
   - AI Services: http://localhost:5000

## 🔄 Development Workflow

### 1. Create a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `test/` - Test improvements
- `refactor/` - Code refactoring

### 2. Make Your Changes

- Write clean, documented code
- Add/update tests
- Update documentation
- Follow coding standards

### 3. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat: add voice emotion detection"

# Sign your commits (required)
git commit -S -m "feat: add voice emotion detection"
```

### 4. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 5. Create a Pull Request

- Use the PR template
- Link related issues
- Add appropriate labels
- Request reviews

## 📝 Coding Standards

### TypeScript/JavaScript

Follow our [TypeScript Style Guide](docs/typescript-style-guide.md):

```typescript
// ✅ Good
export const calculateVoiceQuality = (samples: VoiceSample[]): number => {
  if (samples.length === 0) {
    throw new Error('No voice samples provided');
  }
  
  return samples.reduce((sum, sample) => sum + sample.quality, 0) / samples.length;
};

// ❌ Bad
export function calc_quality(s) {
  return s.reduce((sum, sample) => sum + sample.quality, 0) / s.length;
}
```

Key standards:
- Use TypeScript for all new code
- Functional programming preferred
- Clear, descriptive names
- Comprehensive error handling
- JSDoc comments for public APIs

### Python

Follow our [Python Style Guide](docs/python-style-guide.md):

```python
# ✅ Good
def process_voice_sample(
    audio_data: np.ndarray,
    sample_rate: int = 16000
) -> VoiceFeatures:
    """Process raw audio data and extract voice features.
    
    Args:
        audio_data: Raw audio samples as numpy array
        sample_rate: Audio sample rate in Hz
        
    Returns:
        Extracted voice features
        
    Raises:
        ValueError: If audio_data is empty or invalid
    """
    if len(audio_data) == 0:
        raise ValueError("Audio data cannot be empty")
        
    # Processing logic...
    return features

# ❌ Bad
def process(data):
    # process audio
    return data
```

Key standards:
- Type hints required
- Google-style docstrings
- PEP 8 compliance
- Comprehensive error handling

### React Components

```tsx
// ✅ Good
interface VoiceRecorderProps {
  onRecordingComplete: (audio: Blob) => void;
  maxDuration?: number;
}

export const VoiceRecorder: FC<VoiceRecorderProps> = ({
  onRecordingComplete,
  maxDuration = 300
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [duration, setDuration] = useState(0);
  
  // Component logic...
  
  return (
    <div className="voice-recorder">
      {/* Component JSX */}
    </div>
  );
};

// ❌ Bad
export function VoiceRecorder(props) {
  // No types, no documentation
  return <div>{/* ... */}</div>;
}
```

### CSS/Styling

- Use Tailwind CSS utilities
- Custom CSS only when necessary
- Follow BEM naming for custom classes
- Mobile-first responsive design

## 💬 Commit Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or modifications
- `chore`: Build process or auxiliary tool changes
- `perf`: Performance improvements

### Examples
```bash
# Feature
feat(voice): add emotion detection to voice synthesis

# Bug fix
fix(auth): resolve token refresh race condition

# Documentation
docs(api): update voice training endpoint documentation

# With breaking change
feat(api)!: change voice sample format to support multiple codecs

BREAKING CHANGE: Voice samples now require codec specification
```

## 🔄 Pull Request Process

### Before Submitting

- [ ] Tests pass locally (`npm test`)
- [ ] Linting passes (`npm run lint`)
- [ ] Build succeeds (`npm run build`)
- [ ] Documentation updated
- [ ] Commits are signed
- [ ] Branch is up to date with main

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex code
- [ ] Documentation updated
- [ ] No security vulnerabilities introduced
```

### Review Process

1. **Automated Checks**: CI/CD runs tests, linting, security scans
2. **Code Review**: At least 2 approvals required
3. **Testing**: QA team tests significant changes
4. **Merge**: Squash and merge to main

## 🧪 Testing Guidelines

### Test Requirements

- Minimum 80% code coverage
- All new features must have tests
- All bug fixes must have regression tests

### Test Structure

```typescript
// Unit Test Example
describe('VoiceProcessor', () => {
  describe('processAudioSample', () => {
    it('should extract features from valid audio', async () => {
      const audio = generateTestAudio();
      const features = await processAudioSample(audio);
      
      expect(features).toBeDefined();
      expect(features.pitch).toBeGreaterThan(0);
      expect(features.energy).toBeGreaterThan(0);
    });
    
    it('should throw error for invalid audio', async () => {
      await expect(processAudioSample([])).rejects.toThrow('Invalid audio data');
    });
  });
});

// Integration Test Example
describe('Voice Training API', () => {
  it('should complete voice training workflow', async () => {
    const samples = await uploadVoiceSamples();
    const trainingId = await startTraining(samples);
    const result = await waitForTraining(trainingId);
    
    expect(result.status).toBe('completed');
    expect(result.modelUrl).toBeDefined();
  });
});
```

### Testing Best Practices

- Test behavior, not implementation
- Use meaningful test descriptions
- Keep tests focused and isolated
- Mock external dependencies
- Use factories for test data

## 📚 Documentation

### Code Documentation

- All public APIs must have JSDoc/docstrings
- Complex algorithms need inline comments
- README files for each service
- Architecture Decision Records (ADRs) for significant decisions

### User Documentation

- Update user guides for new features
- Include screenshots/videos where helpful
- Keep language simple and empathetic
- Test documentation with non-technical users

## 🌐 Community

### Getting Help

- **Discord**: [Join our server](https://discord.gg/ethernalecho)
- **GitHub Discussions**: For general questions
- **Stack Overflow**: Tag with `ethernalecho`

### Staying Updated

- **Blog**: https://blog.ethernalecho.com
- **Twitter**: @ethernalecho
- **Newsletter**: Monthly updates on development

### Recognition

We recognize contributors in our:
- Release notes
- Contributors page
- Annual contributor awards

## 🎉 Thank You!

Your contributions help families preserve precious memories and connections. Every improvement, no matter how small, makes a difference in someone's life.

If you have any questions, please don't hesitate to reach out!

---

*EthernalEcho - Built with Love by Our Community*
