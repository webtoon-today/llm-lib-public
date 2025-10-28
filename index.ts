import { AnthropicAdapter } from './adapters/anthropic';
import { GoogleAdapter } from './adapters/google';
import { OpenAIAdapter } from './adapters/openai';
import { KlingAdapter } from './adapters/kling';
import { VeniceAdapter } from './adapters/venice';
import { XAIAdapter } from './adapters/xai';
import { retryWithBackoff, getApiKey, parseLLMJson, uploadBase64ImageToS3, log, generateTrackId, logTracking, isValidBase64 } from './utils';

// LLM Layer - Unified interface for multiple LLM providers
// Exports: generateText, generateImage, generateStructuredData

// Core interfaces
export type LLMMessage = {
    role: 'user' | 'assistant' | 'system';
    content: string | {
        type: 'text' | 'image';
        text?: string;
        image?: string; // base64
    }[];
}

// Provider types
export type LLMProvider = 'google' | 'anthropic' | 'openai' | 'kling' | 'venice' | 'xai';

// Base request type with common fields
type BaseLLMRequest = {
    model: {
        anthropic?: string;
        google?: string;
        openai?: string;
        kling?: string;
        venice?: string;
        xai?: string;
    };
    fallbackOrder?: LLMProvider[];
    retry?: number | number[];
    errorLevel?: 'quiet' | 'info' | 'warn' | 'error';
    caller?: string;
}

// Text generation request
export type TextGenerationRequest = BaseLLMRequest & {
    system: string;
    messages: LLMMessage[];
    maxToken?: number;
    temperature?: number;
}

// Image generation request
export type ImageGenerationRequest = BaseLLMRequest & {
    prompt: string;
    system?: string; // Optional system prompt for image generation

    referenceImage?: string | string[]; 
    width?: number; // For image width
    height?: number; // For image height
    quality?: 'standard' | 'hd'; // For quality settings
}

// Structured data generation request
export type StructuredDataRequest = BaseLLMRequest & {
    system: string;
    messages: LLMMessage[];
    maxToken?: number;
    temperature?: number;
}

// Stream generation request
export type StreamGenerationRequest = BaseLLMRequest & {
    system: string;
    messages: LLMMessage[];
    maxToken?: number;
    temperature?: number;
}

// Tracking information
export type LLMTracking = {
    trackid: string;
    provider: LLMProvider;
    itoken: number;
    otoken: number;
    ttoken: number;
    started_at: number;
    elapsed_seconds: number;
    model: string;
    request_type: 'text' | 'image' | 'structured' | 'stream';
    caller: string;
    error?: string;
    retry_count?: number;
}

// Response types
export type LLMResponse<T = any> = {
    text?: string;
    imageUrl?: string; 
    data: T;
    provider: LLMProvider;
    model: string;
    usage?: {
        promptTokens?: number;
        completionTokens?: number;
        totalTokens?: number;
    };
}

// Response types
export type StreamLLMResponse<T = any> = {
    text: string;
    imageUrl?: string;
    data?: T;
    provider: LLMProvider;
    model: string;
    streamStatus?: 'error';
    error?: LLMError;
};

// Error types
export class LLMError extends Error {
    constructor(
        message: string,
        public provider: LLMProvider,
        public code?: string,
        public statusCode?: number
    ) {
        super(message);
        this.name = 'LLMError';
    }
}

// Default configurations
const DEFAULT_CONFIG = {
    maxToken: 1024,
    temperature: 0.7,
    retry: 1,
    fallbackOrder: ['google', 'anthropic'] as LLMProvider[]
};

// Model defaults by use case
const MODEL_DEFAULTS = {
    text: {
        google: 'gemini-2.5-flash',
        anthropic: 'claude-3-7-sonnet-latest',
        venice: 'venice-uncensored',
        xai: 'grok-4-fast-non-reasoning'
    },
    image: {
        google: 'gemini-2.0-flash-exp-image-generation',
        openai: 'gpt-image-1',
        kling: 'kling-v1-5',
        venice: 'lustify-sdxl',
        xai: 'grok-2-image',
    },
    structured: {
        google: 'gemini-2.5-flash',
        anthropic: 'claude-3-7-sonnet-latest',
        venice: 'venice-uncensored',
        xai: 'grok-4-fast-non-reasoning'
    },
    vision: {
        google: 'gemini-2.0-flash',
        anthropic: 'claude-3-7-sonnet-latest',
        venice: 'venice-uncensored',
        xai: 'grok-4-fast-non-reasoning'
    }
};

// Main exported functions

let anthropic: AnthropicAdapter;
let google: GoogleAdapter;
let openai: OpenAIAdapter;
let kling: KlingAdapter;
let venice: VeniceAdapter;
let xai: XAIAdapter;

/**
 * Generate text using LLM with automatic fallback
 * @param request Text generation request configuration
 * @returns Promise with generated text
 */
export async function generateText(request: TextGenerationRequest): Promise<LLMResponse> {

    const config = {
        ...DEFAULT_CONFIG,
        ...request,
        model: {
            ...MODEL_DEFAULTS.text,
            ...request.model
        }
    };

    const fallbackOrder = config.fallbackOrder || DEFAULT_CONFIG.fallbackOrder;
    const errorLevel = config.errorLevel || 'quiet';
    let lastError: Error | null = null;
    
    // Generate tracking ID and start timer
    const trackId = generateTrackId();
    const started_at = Date.now();
    const caller = request.caller || '[warning] no caller';
    
    log('info', errorLevel, `Starting generateText with providers: ${fallbackOrder.join(', ')}`);
    
    if (fallbackOrder?.includes('anthropic') && !anthropic) {
        const apiKey = await getApiKey('anthropic');
        anthropic = new AnthropicAdapter(apiKey);
    }
    if (fallbackOrder?.includes('google') && !google) {
        const apiKey = await getApiKey('google');
        google = new GoogleAdapter(apiKey);
    }
    if (fallbackOrder?.includes('openai') && !openai) {
        const apiKey = await getApiKey('openai');
        openai = new OpenAIAdapter(apiKey);
    }
    if (fallbackOrder?.includes('venice') && !venice) {
        const apiKey = await getApiKey('venice');
        venice = new VeniceAdapter(apiKey);
    }
    if (fallbackOrder?.includes('xai') && !xai) {
        const apiKey = await getApiKey('xai');
        xai = new XAIAdapter(apiKey);
    }

    for (const provider of fallbackOrder) {
        const model = config.model[provider];
        if (!model) continue;

        try {
            log('info', errorLevel, `Trying provider: ${provider} with model: ${model}`);
            let adapterResponse: { response: string; usage: { itoken: number; otoken: number; ttoken: number } } | undefined;

            await retryWithBackoff(async () => {
                switch (provider) {
                    case 'anthropic':
                        adapterResponse = await anthropic.generateText( model, config.system, config.messages, config.maxToken, config.temperature );
                        break;
                    case 'google':
                        adapterResponse = await google.generateText( model, config.system, config.messages, config.maxToken, config.temperature );
                        break;
                    case 'openai':
                        adapterResponse = await openai.generateText( model, config.system, config.messages, config.maxToken, config.temperature );
                        break;
                    case 'venice':
                        adapterResponse = await venice.generateText( model, config.system, config.messages, config.maxToken, config.temperature );
                        break;
                    case 'xai':
                        adapterResponse = await xai.generateText( model, config.system, config.messages, config.maxToken, config.temperature );
                        break;
                }
            }, Array.isArray(config.retry) ? config.retry[0] : config.retry || DEFAULT_CONFIG.retry);

            log('info', errorLevel, `Success with provider: ${provider}`);
            
            // Log tracking data
            const finished_at = Date.now();
            if (adapterResponse) {
                logTracking({
                    trackid: trackId,
                    provider,
                    itoken: adapterResponse.usage.itoken,
                    otoken: adapterResponse.usage.otoken,
                    ttoken: adapterResponse.usage.ttoken,
                    started_at,
                    elapsed_seconds: finished_at - started_at,
                    model,
                    request_type: 'text',
                    caller,
                    retry_count: 0 // TODO: track actual retry count
                });
                
                return {
                    text: adapterResponse.response,
                    data: undefined,
                    provider,
                    model
                };
            } else {
                throw new Error('No response from adapter');
            }
        } catch (error: any) {
            lastError = error;
            log('warn', errorLevel, `Failed with ${provider}: ${error.message}`);
        }
    }

    log('error', errorLevel, 'All providers failed', lastError);
    
    // Log failed tracking data for generateText
    const finishedAt = Date.now();
    logTracking({
        trackid: trackId,
        provider: fallbackOrder[0] || 'unknown',
        itoken: 0,
        otoken: 0,  
        ttoken: 0,
        started_at: started_at,
        elapsed_seconds: finishedAt - started_at,
        model: 'unknown',
        request_type: 'text',
        caller,
        error: lastError?.message || 'All providers failed',
        retry_count: 0
    });
    
    throw lastError || new Error('All providers failed');
}

/**
 * Generate image using LLM with automatic fallback
 * @param request Image generation request configuration
 * @returns Promise with image URL
 */
export async function generateImage(request: ImageGenerationRequest): Promise<LLMResponse> {
    const config = {
        ...DEFAULT_CONFIG,
        ...request,
        model: {
            ...MODEL_DEFAULTS.image,
            ...request.model
        }
    };

    const fallbackOrder = config.fallbackOrder || ['google', 'openai', 'kling'];
    const errorLevel = config.errorLevel || 'quiet';
    let lastError: Error | null = null;
    
    // Generate tracking ID and start timer
    const trackId = generateTrackId();
    const startedAt = Date.now();
    const caller = request.caller || '[warning] no caller';

    log('info', errorLevel, `Starting generateImage with providers: ${fallbackOrder.join(', ')}`);

    // Check if Anthropic is in the model config or fallback order
    if (config.model.anthropic || fallbackOrder.includes('anthropic')) {
        throw new Error('Anthropic does not support image generation. Please use Google, OpenAI, or Kling for image generation.');
    }

    // Prepare adapters if needed
    if (fallbackOrder?.includes('google') && !google) {
        const apiKey = await getApiKey('google');
        google = new GoogleAdapter(apiKey);
    }
    if (fallbackOrder?.includes('openai') && !openai) {
        const apiKey = await getApiKey('openai');
        openai = new OpenAIAdapter(apiKey);
    }
    if (fallbackOrder?.includes('kling') && !kling) {
        kling = new KlingAdapter();
    }
    if (fallbackOrder?.includes('venice') && !venice) {
        const apiKey = await getApiKey('venice');
        venice = new VeniceAdapter(apiKey);
    }
    if (fallbackOrder?.includes('xai') && !xai) {
        const apiKey = await getApiKey('xai');
        xai = new XAIAdapter(apiKey);
    }

    // Use prompt directly from the request
    const prompt = config.prompt;

    for (const provider of fallbackOrder) {
        const model = config.model[provider];
        if (!model) continue;

        try {
            log('info', errorLevel, `Trying provider: ${provider} with model: ${model}`);
            let imageUrl: string = '';
            let adapterUsage: { itoken: number; otoken: number; ttoken: number } | undefined;

            // Helper function to handle S3 upload for base64 images
            const handleImageUpload = async (imageUrl: string, providerName: string): Promise<string> => {
                if (imageUrl.startsWith('data:image')) {
                    log('info', errorLevel, `Uploading ${providerName} image to S3...`);
                    const s3Url = await uploadBase64ImageToS3(imageUrl);
                    if (s3Url) {
                        log('info', errorLevel, `Image uploaded to: ${s3Url}`);
                        return s3Url;
                    } else {
                        throw new Error('Failed to upload image to S3');
                    }
                }
                return imageUrl;
            };

            await retryWithBackoff(async () => {
                switch (provider) {
                    case 'google':
                        const googleResult = await google.generateImage({
                            model,
                            prompt,
                            system: config.system,
                            width: config.width,
                            height: config.height,
                            referenceImage: config.referenceImage
                        });
                        adapterUsage = googleResult.usage;
                        imageUrl = await handleImageUpload(googleResult.imageUrl, 'Google');
                        break;
                    case 'openai':
                        const openaiResult = await openai.generateImage({
                            model,
                            prompt,
                            system: config.system,
                            width: config.width,
                            height: config.height,
                            referenceImage: config.referenceImage
                        });
                        adapterUsage = openaiResult.usage;
                        imageUrl = await handleImageUpload(openaiResult.imageUrl, 'OpenAI');
                        break;
                    case 'kling':
                        const klingResult = await kling.generateImage({
                            model,
                            prompt,
                            system: config.system,
                            width: config.width,
                            height: config.height,
                            referenceImage: config.referenceImage
                        });
                        adapterUsage = klingResult.usage;
                        imageUrl = await handleImageUpload(klingResult.imageUrl, 'Kling');
                        break;
                    case 'venice':
                        const veniceResult = await venice.generateImage({
                            model,
                            prompt,
                            system: config.system,
                            width: config.width,
                            height: config.height,
                            referenceImage: config.referenceImage
                        });
                        adapterUsage = veniceResult.usage;
                        imageUrl = await handleImageUpload(veniceResult.imageUrl, 'Venice');
                        break;
                    case 'xai':
                        const xaiResult = await xai.generateImage({
                            model,
                            prompt,
                            system: config.system,
                            width: config.width,
                            height: config.height,
                            referenceImage: config.referenceImage
                        });
                        adapterUsage = xaiResult.usage;
                        imageUrl = await handleImageUpload(xaiResult.imageUrl, 'xAI');
                        break;
                    default:
                        throw new Error(`Image generation not supported for ${provider}`);
                }
            }, Array.isArray(config.retry) ? config.retry[0] : config.retry || DEFAULT_CONFIG.retry);

            log('info', errorLevel, `Success with provider: ${provider}`);
            
            // Log tracking data for image generation
            const finishedAt = Date.now();
            logTracking({
                trackid: trackId,
                provider,
                itoken: adapterUsage?.itoken || 0,
                otoken: adapterUsage?.otoken || 0,
                ttoken: adapterUsage?.ttoken || 0,
                started_at: startedAt,
                elapsed_seconds: finishedAt - startedAt,
                model,
                request_type: 'image',
                caller,
                retry_count: 0
            });
            
            return {
                imageUrl,
                data: undefined,
                provider,
                model
            };
        } catch (error: any) {
            lastError = error;
            log('warn', errorLevel, `Failed with ${provider}: ${error.message}`);
        }
    }

    log('error', errorLevel, 'All providers failed', lastError);
    
    // Log failed tracking data for generateImage
    const finishedAt = Date.now();
    logTracking({
        trackid: trackId,
        provider: fallbackOrder[0] || 'unknown',
        itoken: 0,
        otoken: 0,
        ttoken: 0,
        started_at: startedAt,
        elapsed_seconds: finishedAt - startedAt,
        model: 'unknown',
        request_type: 'image',
        caller,
        error: lastError?.message || 'All providers failed',
        retry_count: 0
    });
    
    throw lastError || new Error('All providers failed');
}

/**
 * Generate structured data using LLM with automatic fallback
 * @param request Structured data generation request configuration
 * @returns Promise with parsed structured data
 */
export async function generateStructuredData<T = any>(request: StructuredDataRequest): Promise<LLMResponse<T>> {
    
    const config = {
        ...DEFAULT_CONFIG,
        ...request,
        model: {
            ...MODEL_DEFAULTS.structured,
            ...request.model
        }
    };

    const fallbackOrder = config.fallbackOrder || DEFAULT_CONFIG.fallbackOrder;
    const errorLevel = config.errorLevel || 'quiet';
    let lastError: Error | null = null;
    
    // Generate tracking ID and start timer
    const trackId = generateTrackId();
    const startedAt = Date.now();
    const caller = request.caller || '[warning] no caller';
    
    // Accumulate tokens from all attempts (including JSON parsing failures)
    let totalTokenUsage = { itoken: 0, otoken: 0, ttoken: 0 };
    let retryCount = 0;

    log('info', errorLevel, `Starting generateStructuredData with providers: ${fallbackOrder.join(', ')}`);

    // Prepare adapters if needed
    if (fallbackOrder?.includes('anthropic') && !anthropic) {
        const apiKey = await getApiKey('anthropic');
        anthropic = new AnthropicAdapter(apiKey);
    }
    if (fallbackOrder?.includes('google') && !google) {
        const apiKey = await getApiKey('google');
        google = new GoogleAdapter(apiKey);
    }
    if (fallbackOrder?.includes('openai') && !openai) {
        const apiKey = await getApiKey('openai');
        openai = new OpenAIAdapter(apiKey);
    }
    if (fallbackOrder?.includes('venice') && !venice) {
        const apiKey = await getApiKey('venice');
        venice = new VeniceAdapter(apiKey);
    }
    if (fallbackOrder?.includes('xai') && !xai) {
        const apiKey = await getApiKey('xai');
        xai = new XAIAdapter(apiKey);
    }

    // Add JSON instruction to the last message
    const messages = [...config.messages];
    
    for (const provider of fallbackOrder) {
        const model = config.model[provider];
        if (!model) continue;

        try {
            log('info', errorLevel, `Trying provider: ${provider} with model: ${model}`);
            let adapterResponse: { response: string; usage: { itoken: number; otoken: number; ttoken: number } } | undefined;
            let data: T | undefined;
            retryCount++;

            await retryWithBackoff(async () => {
                switch (provider) {
                    case 'anthropic':
                        adapterResponse = await anthropic.generateText( model, config.system, messages, config.maxToken, config.temperature );
                        break;
                    case 'google':
                        adapterResponse = await google.generateText( model, config.system, messages, config.maxToken, config.temperature );
                        break;
                    case 'openai':
                        adapterResponse = await openai.generateText( model, config.system, messages, config.maxToken, config.temperature );
                        break;
                    case 'venice':
                        adapterResponse = await venice.generateText( model, config.system, messages, config.maxToken, config.temperature );
                        break;
                    case 'xai':
                        adapterResponse = await xai.generateText( model, config.system, messages, config.maxToken, config.temperature );
                        break;
                }

                if (adapterResponse) {
                    // Log this individual attempt immediately (before JSON parsing)
                    const attemptFinishedAt = Date.now();
                    logTracking({
                        trackid: trackId,
                        provider,
                        itoken: adapterResponse.usage.itoken,
                        otoken: adapterResponse.usage.otoken,
                        ttoken: adapterResponse.usage.ttoken,
                        started_at: startedAt,
                        elapsed_seconds: attemptFinishedAt - startedAt,
                        model,
                        request_type: 'structured',
                        caller,
                        retry_count: retryCount - 1,
                        error: undefined // LLM succeeded, but JSON parsing might fail
                    });
                    
                    // Accumulate for potential final summary log
                    totalTokenUsage.itoken += adapterResponse.usage.itoken;
                    totalTokenUsage.otoken += adapterResponse.usage.otoken;
                    totalTokenUsage.ttoken += adapterResponse.usage.ttoken;
                    
                    // Parse the JSON response (this can fail and cause retry)
                    log('info', errorLevel, 'Parsing JSON response...');
                    data = parseLLMJson<T>(adapterResponse.response);
                }
            }, Array.isArray(config.retry) ? config.retry[0] : config.retry || DEFAULT_CONFIG.retry);

            log('info', errorLevel, `Success with provider: ${provider}`);
            
            if (adapterResponse && data) {
                return {
                    text: adapterResponse.response,
                    data: data,
                    provider,
                    model
                };
            } else {
                throw new Error('No response from adapter or failed to parse data');
            }
        } catch (error: any) {
            lastError = error;
            log('warn', errorLevel, `Failed with ${provider}: ${error.message}`);
        }
    }

    log('error', errorLevel, 'All providers failed', lastError);
    throw lastError || new Error('All providers failed');
}

/**
 * Generate streaming text using LLM with automatic fallback
 * @param request Stream generation request configuration
 * @returns AsyncGenerator yielding text chunks
 */
export async function* generateStream(request: StreamGenerationRequest): AsyncGenerator<StreamLLMResponse> {
    const config = {
        ...DEFAULT_CONFIG,
        ...request,
        model: {
            ...MODEL_DEFAULTS.text,
            ...request.model
        }
    };

    const fallbackOrder = config.fallbackOrder || DEFAULT_CONFIG.fallbackOrder;
    const errorLevel = config.errorLevel || 'quiet';
    let lastError: Error | null = null;
    
    // Generate tracking ID and start timer
    const trackId = generateTrackId();
    const startedAt = Date.now();
    const caller = request.caller || '[warning] no caller';
    
    log('info', errorLevel, `Starting generateStream with providers: ${fallbackOrder.join(', ')}`);
    
    // Prepare adapters if needed
    if (fallbackOrder?.includes('anthropic') && !anthropic) {
        const apiKey = await getApiKey('anthropic');
        anthropic = new AnthropicAdapter(apiKey);
    }
    if (fallbackOrder?.includes('google') && !google) {
        const apiKey = await getApiKey('google');
        google = new GoogleAdapter(apiKey);
    }
    if (fallbackOrder?.includes('openai') && !openai) {
        const apiKey = await getApiKey('openai');
        openai = new OpenAIAdapter(apiKey);
    }
    if (fallbackOrder?.includes('venice') && !venice) {
        const apiKey = await getApiKey('venice');
        venice = new VeniceAdapter(apiKey);
    }
    if (fallbackOrder?.includes('xai') && !xai) {
        const apiKey = await getApiKey('xai');
        xai = new XAIAdapter(apiKey);
    }

    // Filter out providers that don't support streaming
    const streamingProviders = fallbackOrder.filter(p => p !== 'kling');
    
    for (const provider of streamingProviders) {
        const model = config.model[provider];
        if (!model) continue;

        try {
            log('info', errorLevel, `Trying streaming with provider: ${provider} using model: ${model}`);
            
            let fullText = '';
            let streamUsage: { itoken: number; otoken: number; ttoken: number } | undefined;
            
            // Create the appropriate stream based on provider
            const streamGenerator = (async function* () {
                try {
                    switch (provider) {
                        case 'anthropic':
                            yield* anthropic.generateStream(
                                model, 
                                config.system, 
                                config.messages, 
                                config.maxToken, 
                                config.temperature
                            );
                            break;
                        case 'google':
                            yield* google.generateStream(
                                model, 
                                config.system, 
                                config.messages, 
                                config.maxToken, 
                                config.temperature
                            );
                            break;
                        case 'openai':
                            yield* openai.generateStream(
                                model,
                                config.system,
                                config.messages,
                                config.maxToken,
                                config.temperature
                            );
                            break;
                        case 'venice':
                            yield* venice.generateStream(
                                model,
                                config.system,
                                config.messages,
                                config.maxToken,
                                config.temperature
                            );
                            break;
                        case 'xai':
                            yield* xai.generateStream(
                                model,
                                config.system,
                                config.messages,
                                config.maxToken,
                                config.temperature
                            );
                            break;
                        default:
                            throw new Error(`Streaming not supported for ${provider}`);
                    }
                } catch (error) {
                    throw error;
                }
            })();

            // Yield chunks as they come
            try {
                for await (const chunk of streamGenerator) {
                    if (chunk.type === 'text' && chunk.text) {
                        fullText += chunk.text;
                        yield {
                            text: chunk.text,
                            imageUrl: undefined,
                            data: undefined,
                            provider,
                            model,
                        };
                    } else if (chunk.type === 'usage' && chunk.usage) {
                        streamUsage = chunk.usage;
                    }
                }
                
                log('info', errorLevel, `Stream completed successfully with provider: ${provider}`);
                
                // Log tracking data for successful streaming
                const finishedAt = Date.now();
                logTracking({
                    trackid: trackId,
                    provider,
                    itoken: streamUsage?.itoken || 0,
                    otoken: streamUsage?.otoken || 0,
                    ttoken: streamUsage?.ttoken || 0,
                    started_at: startedAt,
                    elapsed_seconds: finishedAt - startedAt,
                    model,
                    request_type: 'stream',
                    caller,
                    retry_count: 0
                });
                
                return; // Success - exit the function
                
            } catch (streamError) {
                // If error occurred during streaming, try next provider
                lastError = streamError instanceof Error ? streamError : new Error(String(streamError));
                log('warn', errorLevel, `Stream failed with ${provider}: ${lastError.message}`);
                
                // Yield error response for this provider
                yield {
                    text: fullText,
                    imageUrl: undefined,
                    data: undefined,
                    provider,
                    model,
                    streamStatus: 'error' as const,
                    error: new LLMError(
                        lastError.message,
                        provider,
                        (streamError as LLMError)?.code,
                        (streamError as LLMError)?.statusCode
                    )
                };
                continue; // Try next provider
            }
            
        } catch (error) {
            lastError = error instanceof Error ? error : new Error(String(error));
            log('warn', errorLevel, `Failed to initialize stream with ${provider}: ${lastError.message}`);
        }
    }

    // All providers failed
    log('error', errorLevel, 'All streaming providers failed', lastError);
    yield {
        text: '',
        imageUrl: undefined,
        data: undefined,
        provider: streamingProviders[0] || 'google',
        model: config.model[streamingProviders[0]] || 'unknown',
        streamStatus: 'error' as const,
        error: new LLMError(
            lastError?.message || 'All streaming providers failed',
            streamingProviders[0] || 'google',
            'STREAM_FAILED'
        )
    };
}
