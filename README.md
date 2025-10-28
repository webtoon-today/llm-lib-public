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
npm install https://github.com/webtoon-today/llm-lib-public
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

#### Venice AI
```bash
VENICE_API_KEY=your_venice_api_key
```

#### xAI
```bash
XAI_API_KEY=your_xai_api_key
```

#### Kling AI
```bash
KLING_ACCESS_KEY_ID=your_kling_access_key_id
KLING_ACCESS_KEY_SECRET=your_kling_access_key_secret
```

#### Google Vertex AI
```bash
VERTEX_AI_CREDENTIALS={"type":"service_account","project_id":"..."}
VERTEX_AI_PROJECT=your-gcp-project-id
VERTEX_AI_LOCATION=us-central1  # Optional, defaults to us-central1
```

## Usage

### Text Generation

```typescript
import { generateText } from 'llm-lib-public';

const result = await generateText({
    model: {
        anthropic: 'claude-3-5-haiku-latest',
        google: 'gemini-2.5-flash'
    },
    system: 'You are a helpful assistant.',
    messages: [
        { role: 'user', content: 'Hello!' }
    ],
    maxToken: 1000,
    temperature: 0.7,
    fallbackOrder: ['google', 'anthropic']
});

console.log(result.text);
```

### Text Generation with Image Context (Vision)

```typescript
import { generateText } from 'llm-lib-public';
import axios from 'axios';

// Download image from URL and convert to base64
const imageUrl = 'https://example.com/image.jpg';
const imageResponse = await axios.get(imageUrl, { responseType: 'arraybuffer' });
const imageBase64 = Buffer.from(imageResponse.data).toString('base64');

const result = await generateText({
    model: {
        google: 'gemini-2.5-flash',
        anthropic: 'claude-sonnet-4-5'
    },
    system: 'You are an AI assistant that can analyze images and answer questions about them.',
    messages: [
        {
            role: 'user',
            content: [
                { type: 'text', text: 'What do you see in this image? Describe it in detail.' },
                { type: 'image', image: imageBase64 }
            ]
        }
    ],
    maxToken: 500,
    temperature: 0.7,
    fallbackOrder: ['google', 'anthropic']
});

console.log(result.text);
```

### Streaming Text Generation

```typescript
import { generateStream } from 'llm-lib-public';

const stream = generateStream({
    model: {
        google: 'gemini-2.5-flash',
        anthropic: 'claude-sonnet-4-5'
    },
    system: 'You are a helpful assistant.',
    messages: [
        { role: 'user', content: 'Tell me a story.' }
    ],
    fallbackOrder: ['google', 'anthropic'],
    caller: 'my-app'
});
for await (const chunk of stream) {
    if (chunk.streamStatus === 'streaming') {
        process.stdout.write(chunk.text || '');
    }
}
```

### Structured Data Generation

```typescript
import { generateStructuredData } from 'llm-lib-public';

type Person = {
    name: string;
    age: number;
    city: string;
}

const result = await generateStructuredData<Person>({
    model: {
        google: 'gemini-2.5-flash',
        anthropic: 'claude-haiku-4-5'
    },
    system: `Generate JSON data based on the user request.
        |RETURN FORMAT: Json object
        |{
        |    "name": string,
        |    "age": number,
        |    "city": string | undefined // return undefined if it is not given
        |}
        |`.replace(/\n +\|/g, '\n'),
    messages: [
        { role: 'user', content: 'Create a person who lives in Seoul, 25 years old, and is called Mike.' },
        {
            role: 'assistant',
            content: JSON.stringify({
                "name": "Mike",
                "age": 25,
                "city": "Seoul, Republic of Korea"
            }, undefined, 4),
        },
        { role: 'user', content: 'Create a person with name John, age 30, city NYC' }
    ],
    fallbackOrder: ['google', 'anthropic']
});

console.log(result.data); // { name: "John", age: 30, city: "NYC" }
```

### Image Generation

```typescript
import { generateImage } from 'llm-lib-public';

const result = await generateImage({
    model: {
        google: 'gemini-2.5-flash-image-preview',
        openai: 'gpt-image-1'
    },
    prompt: 'A beautiful sunset over mountains',
    width: 1024,
    height: 768,
    fallbackOrder: ['google', 'openai']
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
