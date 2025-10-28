import OpenAI, {toFile} from 'openai';
import axios from 'axios';
import { LLMMessage, LLMError, LLMProvider } from '../index';
import { ImageGenerationOptions, AdapterResponse, StreamChunk } from '../types';
import { ImageEditParams } from 'openai/resources/images';
import { Uploadable } from 'openai/uploads';
import { isValidBase64 } from '../utils';

export class OpenAIAdapter {
    private client: OpenAI;
    private provider: LLMProvider = 'openai';

    constructor(apiKey: string, baseURL?: string) {
        this.client = new OpenAI({
            apiKey,
            baseURL: baseURL || undefined
        });
        if (baseURL?.includes('venice')) {
            this.provider = 'venice';
        } else if (baseURL?.includes('x.ai')) {
            this.provider = 'xai';
        }
    }

    async generateText(
        model: string,
        system: string,
        messages: LLMMessage[],
        maxTokens: number = 4096,
        temperature: number = 0.7,
        otherOptions: object = {}
    ): Promise<AdapterResponse> {
        try {
            const openaiMessages = this.convertMessages(messages, system);
            
            const response = await this.client.chat.completions.create({
                model,
                messages: openaiMessages,
                max_tokens: maxTokens,
                temperature,
                ...otherOptions
            });

            const text = response.choices[0]?.message?.content || '';
            
            return {
                response: text,
                usage: {
                    itoken: response.usage?.prompt_tokens || 0,
                    otoken: response.usage?.completion_tokens || 0,
                    ttoken: response.usage?.completion_tokens_details?.reasoning_tokens || 0 // OpenAI's thinking tokens for reasoning models
                }
            };
        } catch (error: any) {
            throw new LLMError(
                error.message || 'OpenAI API error',
                this.provider,
                error.code,
                error.status
            );
        }
    }

    async *generateStream(
        model: string,
        system: string,
        messages: LLMMessage[],
        maxTokens: number = 4096,
        temperature: number = 0.7
    ): AsyncGenerator<StreamChunk> {
        try {
            const openaiMessages = this.convertMessages(messages, system);
            
            const stream = await this.client.chat.completions.create({
                model,
                messages: openaiMessages,
                max_tokens: maxTokens,
                temperature,
                stream: true,
                stream_options: { include_usage: true } // Enable usage data in streaming
            });

            for await (const chunk of stream) {
                const content = chunk.choices[0]?.delta?.content;
                
                // Yield text content chunks
                if (content) {
                    yield {
                        type: 'text',
                        text: content
                    };
                }
                
                // Final chunk contains usage data
                if (chunk.usage) {
                    yield {
                        type: 'usage',
                        usage: {
                            itoken: chunk.usage.prompt_tokens || 0,
                            otoken: chunk.usage.completion_tokens || 0,
                            ttoken: chunk.usage.completion_tokens_details?.reasoning_tokens || 0
                        }
                    };
                }
            }
        } catch (error: any) {
            throw new LLMError(
                error.message || 'OpenAI API error',
                this.provider,
                error.code,
                error.status
            );
        }
    }

    async generateImage(options: ImageGenerationOptions): Promise<{ imageUrl: string; usage: { itoken: number; otoken: number; ttoken: number } }> {
        const { model, prompt, system, width, height, referenceImage } = options;
        
        try {
            
            if (system) {
                // ignore system prompt for image generation
            }
            
            // Check if we should use image edit endpoint
            if (referenceImage) {
                // Use images.edit endpoint for image editing
                const editParams: ImageEditParams & {image: Uploadable[]} = {
                    model: model || 'gpt-image-1',
                    prompt,
                    n: 1,
                    image: [] // to be filled
                };
                
                // Only gpt-image-1 supports image editing
                if (model && model !== 'gpt-image-1') {
                    throw new Error(`Image editing is only supported for gpt-image-1, not ${model}`);
                }

                let references = Array.isArray(referenceImage) ? referenceImage : [referenceImage];
                
                for (let refImg of references) {
                    // Handle reference image format
                    if (refImg.startsWith('data:image')) {
                        // Convert data URL to Buffer for the API
                        const base64Data = refImg.split(',')[1];
                        const mimeType = refImg.match(/data:(image\/[a-zA-Z]+);base64,/);
                        editParams.image.push(await toFile(Buffer.from(base64Data, 'base64'), null, {type: mimeType ? mimeType[1] : "image/png"}));
                    } else {
                        editParams.image.push(await toFile(Buffer.from(refImg, 'base64'), null, {type: "image/png"}));
                    }
                }
                
                // Handle size for image editing
                if (width && height) {
                    editParams.size = `${width}x${height}` as ImageEditParams['size'];
                }
                
                const response = await this.client.images.edit(editParams);
                
                // Response contains base64 data
                const b64_json = response.data?.[0]?.b64_json;
                if (!b64_json) {
                    throw new Error('No base64 data in response');
                }
                return {
                    imageUrl: `data:image/png;base64,${b64_json}`,
                    usage: { itoken: 0, otoken: 0, ttoken: 0 } // OpenAI image edit API doesn't provide token usage
                };
            }
            
            // Regular image generation (no reference image)
            const params: any = {
                model: model || 'gpt-image-1',
                prompt,
                n: 1
            };

            // Handle different models and sizes
            if (model === 'gpt-image-1') {
                // gpt-image-1 always returns base64
                if (width && height) {
                    params.size = `${width}x${height}`;
                } else {
                    params.size = 'auto';
                }
                params.quality = 'auto';
                // Note: response_format is not supported for gpt-image-1
            } else if (model === 'dall-e-3') {
                // DALL-E 3 supports: 1024x1024, 1792x1024, or 1024x1792
                if (width && height) {
                    // Find nearest supported size
                    const supportedSizes = [
                        { w: 1024, h: 1024 },
                        { w: 1792, h: 1024 },
                        { w: 1024, h: 1792 }
                    ];
                    
                    let nearestSize = supportedSizes[0];
                    let minDiff = Math.abs(width - nearestSize.w) + Math.abs(height - nearestSize.h);
                    
                    for (const size of supportedSizes) {
                        const diff = Math.abs(width - size.w) + Math.abs(height - size.h);
                        if (diff < minDiff) {
                            minDiff = diff;
                            nearestSize = size;
                        }
                    }
                    
                    params.size = `${nearestSize.w}x${nearestSize.h}`;
                } else {
                    params.size = '1024x1024';
                }
                params.quality = 'standard';
                params.style = 'natural'; // Can be 'natural' or 'vivid'
                params.response_format = 'b64_json'; // Request base64 for consistency
            } else if (model === 'dall-e-2') {
                // DALL-E 2 supports: 256x256, 512x512, or 1024x1024
                if (width && height) {
                    // Find nearest supported size
                    const supportedSizes = [
                        { w: 256, h: 256 },
                        { w: 512, h: 512 },
                        { w: 1024, h: 1024 }
                    ];
                    
                    let nearestSize = supportedSizes[0];
                    let minDiff = Math.abs(width - nearestSize.w) + Math.abs(height - nearestSize.h);
                    
                    for (const size of supportedSizes) {
                        const diff = Math.abs(width - size.w) + Math.abs(height - size.h);
                        if (diff < minDiff) {
                            minDiff = diff;
                            nearestSize = size;
                        }
                    }
                    
                    params.size = `${nearestSize.w}x${nearestSize.h}`;
                } else {
                    params.size = '1024x1024';
                }
                params.response_format = 'b64_json'; // Request base64 for consistency
            }

            const response = await this.client.images.generate(params);

            // Handle response based on format
            if (model === 'gpt-image-1' || params.response_format === 'b64_json') {
                // Response contains base64 data
                const b64_json = response.data?.[0]?.b64_json;
                if (!b64_json) {
                    throw new Error('No base64 data in response');
                }
                return {
                    imageUrl: `data:image/png;base64,${b64_json}`,
                    usage: { itoken: 0, otoken: 0, ttoken: 0 } // OpenAI image generation API doesn't provide token usage in response
                };
            } else {
                // Response contains URL (shouldn't happen with our config, but handle it)
                const imageUrl = response.data?.[0]?.url;
                if (!imageUrl) {
                    throw new Error('No image URL in response');
                }
                
                // Download and convert to base64
                const imageResponse = await axios.get(imageUrl, {
                    responseType: 'arraybuffer'
                });
                const base64 = Buffer.from(imageResponse.data).toString('base64');
                return {
                    imageUrl: `data:image/png;base64,${base64}`,
                    usage: { itoken: 0, otoken: 0, ttoken: 0 } // OpenAI image generation API doesn't provide token usage in response
                };
            }
        } catch (error: any) {
            throw new LLMError(
                error.message || 'OpenAI API error',
                this.provider,
                error.code,
                error.status
            );
        }
    }

    private convertMessages(messages: LLMMessage[], system: string): OpenAI.Chat.ChatCompletionMessageParam[] {
        const openaiMessages: OpenAI.Chat.ChatCompletionMessageParam[] = [];
        
        if (system) {
            openaiMessages.push({ role: 'system', content: system });
        }

        messages.forEach(msg => {
            if (typeof msg.content === 'string') {
                openaiMessages.push({
                    role: msg.role as 'user' | 'assistant',
                    content: msg.content
                });
            } else {
                // Handle multimodal content
                const content: any[] = msg.content.map(c => {
                    if (c.type === 'text') {
                        return { type: 'text', text: c.text || '' };
                    } else if (c.type === 'image') {
                        return {
                            type: 'image_url',
                            image_url: {
                                url: `data:image/png;base64,${c.image}`
                            }
                        };
                    }
                    return { type: 'text', text: '' };
                });

                openaiMessages.push({
                    role: msg.role as 'user' | 'assistant',
                    content
                });
            }
        });

        return openaiMessages;
    }
}