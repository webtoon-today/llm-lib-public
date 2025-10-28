# AI LLM Library

A TypeScript library for interacting with multiple AI/LLM providers with automatic fallback support.

## Supported Providers

- **Anthropic** (Claude)
- **Google AI** (Gemini)
- **OpenAI** (GPT, DALL-E)
- **Venice AI**
- **xAI**
- **Kling AI** (Image generation)
- **Google Vertex AI** (Optional, for Vertex AI models)

## Installation

```bash
npm install
```

## Configuration

All API keys and credentials are provided via environment variables. You only need to set the environment variables for the providers you plan to use.

### Required Environment Variables (per provider)

#### Anthropic
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
```

#### Google AI
```bash
GOOGLE_AI_API_KEY=your_google_ai_api_key
```

#### OpenAI
```bash
OPENAI_API_KEY=your_openai_api_key
```

#### Venice AI (Optional)
```bash
VENICE_API_KEY=your_venice_api_key
```

#### xAI (Optional)
```bash
XAI_API_KEY=your_xai_api_key
```

#### Kling AI (Optional)
```bash
KLING_ACCESS_KEY_ID=your_kling_access_key_id
KLING_ACCESS_KEY_SECRET=your_kling_access_key_secret
```

#### Google Vertex AI (Optional)
```bash
VERTEX_AI_CREDENTIALS={"type":"service_account","project_id":"..."}
VERTEX_AI_PROJECT=your-gcp-project-id
VERTEX_AI_LOCATION=us-central1  # Optional, defaults to us-central1
```

## Usage

### Text Generation

```typescript
import { generateText } from './index';

const result = await generateText({
    model: {
        anthropic: 'claude-3-haiku-20240307',
        google: 'gemini-1.5-flash'
    },
    system: 'You are a helpful assistant.',
    messages: [
        { role: 'user', content: 'Hello!' }
    ],
    maxToken: 1000,
    temperature: 0.7,
    fallbackOrder: ['anthropic', 'google']
});

console.log(result.text);
```

### Streaming Text Generation

```typescript
import { generateStream } from './index';

for await (const chunk of generateStream({
    model: { anthropic: 'claude-3-haiku-20240307' },
    system: 'You are a helpful assistant.',
    messages: [
        { role: 'user', content: 'Tell me a story.' }
    ],
    fallbackOrder: ['anthropic'],
    caller: 'my-app'
})) {
    if (chunk.streamStatus === 'streaming') {
        process.stdout.write(chunk.text || '');
    }
}
```

### Structured Data Generation

```typescript
import { generateStructuredData } from './index';

interface Person {
    name: string;
    age: number;
    city: string;
}

const result = await generateStructuredData<Person>({
    model: { google: 'gemini-1.5-flash' },
    system: 'Generate JSON data based on the user request.',
    messages: [
        { role: 'user', content: 'Create a person with name John, age 30, city NYC' }
    ],
    fallbackOrder: ['google']
});

console.log(result.data); // { name: "John", age: 30, city: "NYC" }
```

### Image Generation

```typescript
import { generateImage } from './index';

const result = await generateImage({
    model: { google: 'gemini-2.0-flash-exp-image-generation' },
    prompt: 'A beautiful sunset over mountains',
    width: 1024,
    height: 768,
    fallbackOrder: ['google']
});

console.log(result.imageUrl); // Returns base64 data URL
```

## Features

- **Automatic Fallback**: If a provider fails, automatically tries the next provider in the fallback order
- **Retry Logic**: Built-in retry mechanism with exponential backoff
- **Multimodal Support**: Support for text and image inputs
- **Streaming**: Real-time streaming support for text generation
- **Type Safety**: Full TypeScript support with comprehensive types
- **Error Handling**: Detailed error messages with provider-specific error codes
- **Logging**: Configurable logging levels (quiet, info, warn, error)
- **Usage Tracking**: Track token usage across all providers

## Testing

```bash
npm test
```

Make sure to set up the required environment variables before running tests.

## Notes

- API keys are only required for the providers you actually use
- If a provider is not configured (missing API key), it will fail gracefully and try the next provider in the fallback order
- Image generation returns base64 data URLs (no cloud storage required)
- Logging is done to console only (no cloud logging services)

## License

ISC
