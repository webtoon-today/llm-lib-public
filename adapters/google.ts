import { GoogleGenAI, Content, Modality } from '@google/genai';
import { LLMMessage, LLMError } from '../index';
import { ImageGenerationOptions, AdapterResponse, StreamChunk } from '../types';

export class GoogleAdapter {
    private client: GoogleGenAI;
    private clientVertexai: GoogleGenAI|null = null;

    constructor(apiKey: string) {
        this.client = new GoogleGenAI({ apiKey });
    }

    async initVertexai () {
        if (this.clientVertexai) {
            return; // Already initialized
        }

        const vertexAuthJson = process.env.VERTEX_AI_CREDENTIALS;
        const project = process.env.VERTEX_AI_PROJECT;
        const location = process.env.VERTEX_AI_LOCATION || 'us-central1';

        if (!vertexAuthJson || !project) {
            throw new LLMError(
                'Google Vertex AI credentials not found. Please set VERTEX_AI_CREDENTIALS and VERTEX_AI_PROJECT environment variables.',
                'google',
                'MISSING_CREDENTIALS'
            );
        }

        try {
            const credentials = JSON.parse(vertexAuthJson);

            this.clientVertexai = new GoogleGenAI({
                vertexai: true,
                location,
                project,
                googleAuthOptions: { credentials }
            });
        } catch (error: any) {
            throw new LLMError(`Failed to initialize Vertex AI: ${error.message}`, 'google', 'INIT_FAILED');
        }
    }

    async generateText(
        model: string,
        system: string,
        messages: LLMMessage[],
        maxTokens: number = 4096,
        temperature: number = 0.7
    ): Promise<AdapterResponse> {
        try {
            const session = this.client.chats.create({
                model,
                config: {
                    responseModalities: ['text'],
                    systemInstruction: system,
                },
                history: this.convertMessagesToHistory(messages.slice(0, -1))
            });

            // Send the last message
            const lastMessage = messages[messages.length - 1];
            const response = await session.sendMessage({
                message: this.convertMessageToParts(lastMessage)
            });

            const text = response.candidates?.[0]?.content?.parts?.[0]?.text ?? '';
            
            return {
                response: text,
                usage: {
                    itoken: response.usageMetadata?.promptTokenCount || 0,
                    otoken: response.usageMetadata?.candidatesTokenCount || 0,
                    ttoken: response.usageMetadata?.thoughtsTokenCount || 0
                }
            };
        } catch (error: any) {
            throw new LLMError(
                error.message || 'Google AI API error',
                'google',
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
            const session = this.client.chats.create({
                model,
                config: {
                    responseModalities: ['text'],
                    systemInstruction: system,
                },
                history: this.convertMessagesToHistory(messages.slice(0, -1))
            });

            // Send the last message with streaming
            const lastMessage = messages[messages.length - 1];
            const stream = await session.sendMessageStream({
                message: this.convertMessageToParts(lastMessage)
            });

            let lastEvent: any = null;

            for await (const event of stream) {
                if (event.text) {
                    yield {
                        type: 'text',
                        text: event.text
                    };
                }
                lastEvent = event; // Keep track of the last event
            }

            // The final event should contain usageMetadata
            if (lastEvent?.usageMetadata) {
                yield {
                    type: 'usage',
                    usage: {
                        itoken: lastEvent.usageMetadata.promptTokenCount || 0,
                        otoken: lastEvent.usageMetadata.candidatesTokenCount || 0,
                        ttoken: lastEvent.usageMetadata.thoughtsTokenCount || 0
                    }
                };
            }
        } catch (error: any) {
            throw new LLMError(
                error.message || 'Google AI API error',
                'google',
                error.code,
                error.status
            );
        }
    }

    async generateImage(options: ImageGenerationOptions): Promise<{ imageUrl: string; usage: { itoken: number; otoken: number; ttoken: number } }> {
        const { model, prompt, system, width, height, referenceImage } = options;
        try {

            if (model.startsWith('image')) {
                return this.generateImageVertex(options);
            }
            // Special handling for Gemini image generation model
            if (model.includes('image')) {
                const session = this.client.chats.create({
                    model,
                    config: {
                        responseModalities: [Modality.TEXT, Modality.IMAGE],
                    }
                });

                let message: any[] = [];
                
                if (referenceImage) {

                    const references = Array.isArray(referenceImage) ? referenceImage : [referenceImage];

                    message = [
                        {
                            text: prompt,
                        },
                    ];
                    
                    for (const refImg of references) {
                        // Image editing mode - erase text from image
                        const imageData = refImg.startsWith('data:') 
                            ? refImg.split(',')[1] 
                            : refImg;
                        
                        const mimeType = refImg.match(/data:(image\/[a-zA-Z]+);base64,/);
                        
                        message.push({
                            inlineData: {
                                data: imageData,
                                mimeType: mimeType ? mimeType[1] : 'image/png'
                            }
                        });
                    }
                } else {
                    // Image generation mode
                    message = [{
                        text: `
                            |${system ?? `Hi, can you create a simple illustration based on the following description?
                                |In any case, never include text in the image. If image has text, it will be rejected.
                                |Please create a clean, simple image without any text or writing.
                                |`.replace(/\n *\|/g, '\n')}\n\n
                            |Description: ${prompt}
                            |`.replace(/\n *\|/g, '\n'),
                    }];
                }

                const response = await session.sendMessage({ message });
                
                // Extract base64 image data from response
                for (const part of response.candidates?.[0]?.content?.parts ?? []) {
                    if (part.inlineData?.data) {
                        // Return base64 data with proper prefix and usage metadata
                        return {
                            imageUrl: `data:image/png;base64,${part.inlineData.data}`,
                            usage: {
                                itoken: response.usageMetadata?.promptTokenCount || 0,
                                otoken: response.usageMetadata?.candidatesTokenCount || 0,
                                ttoken: response.usageMetadata?.thoughtsTokenCount || 0
                            }
                        };
                    }
                }
                
                throw new Error('No image data in response');
            } else {
                throw new Error(`Image generation not supported for model: ${model}`);
            }
        } catch (error: any) {
            throw new LLMError(
                error.message || 'Google AI API error',
                'google',
                error.code,
                error.status
            );
        }
    }

    async generateImageVertex(options: ImageGenerationOptions): Promise<{ imageUrl: string; usage: { itoken: number; otoken: number; ttoken: number } }> {

        await this.initVertexai();

        const { width, height } = options;

        // Calculate aspect ratio from width and height
        let aspectRatio = "1:1"; // Default
        
        if (width && height) {
            // Allowed aspect ratios for Kling
            const allowedRatios = [
                { ratio: "16:9", value: 16/9 },
                { ratio: "9:16", value: 9/16 },
                { ratio: "1:1", value: 1 },
                { ratio: "4:3", value: 4/3 },
                { ratio: "3:4", value: 3/4 },
                { ratio: "3:2", value: 3/2 },
                { ratio: "2:3", value: 2/3 },
                { ratio: "21:9", value: 21/9 }
            ];
            
            const inputRatio = width / height;
            
            // Find the nearest allowed ratio
            let nearestRatio = allowedRatios[0];
            let minDiff = Math.abs(inputRatio - nearestRatio.value);
            
            for (const allowed of allowedRatios) {
                const diff = Math.abs(inputRatio - allowed.value);
                if (diff < minDiff) {
                    minDiff = diff;
                    nearestRatio = allowed;
                }
            }
            
            aspectRatio = nearestRatio.ratio;
        }

        if (options.referenceImage) {
            console.warn('Reference image is not supported for Vertex AI image generation');
        }

        const res = await this.clientVertexai!.models.generateImages({
            model: options.model,
            prompt: options.prompt,
            config: { aspectRatio, outputMimeType: 'image/png' }
        })

        let bytes = res.generatedImages?.[0].image?.imageBytes;

        if (!bytes) {
            throw new Error('No image data in response');
        }

        return {
            imageUrl: `data:image/png;base64,${bytes}`,
            usage: { itoken: 0, otoken: 0, ttoken: 0 } // Vertex AI doesn't provide token usage for image gen
        };
            
    }

    private convertMessagesToHistory(messages: LLMMessage[]): Content[] {
        return messages.map(msg => ({
            role: msg.role === 'user' ? 'user' : 'model',
            parts: this.convertMessageToParts(msg)
        }));
    }

    private convertMessageToParts(message: LLMMessage): any[] {
        if (typeof message.content === 'string') {
            return [{ text: message.content }];
        }
        
        return message.content.map(c => {
            if (c.type === 'text') {
                return { text: c.text || '' };
            } else if (c.type === 'image') {
                return { 
                    inlineData: {
                        data: c.image || '',
                        mimeType: 'image/png'
                    }
                };
            }
            return { text: '' };
        });
    }
}