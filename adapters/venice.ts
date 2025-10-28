import { OpenAIAdapter } from './openai';
import { ImageGenerationOptions, AdapterUsage, AdapterResponse, StreamChunk } from '../types';
import { LLMError, LLMMessage } from '../index';

interface VeniceImageRequest {
    model: string;
    prompt: string;
    cfg_scale?: number;
    width?: number;
    height?: number;
    variants?: number;
    negative_prompt?: string;
    steps?: number;
    seed?: number;
    style_preset?: string;
    safe_mode?: boolean;
    format?: string;
}

interface VeniceImageEditRequest {
    prompt: string;
    image: string;  // base64 encoded string
}

interface VeniceImageResponse {
    id: string;
    images: string[];
    timing?: {
        inference?: number;
        upload?: number;
    };
    request?: VeniceImageRequest;
}

export class VeniceAdapter extends OpenAIAdapter {
    private veniceBaseUrl = 'https://api.venice.ai/api/v1';
    private apiKey: string;

    constructor(apiKey: string) {
        super(apiKey, 'https://api.venice.ai/api/v1');
        this.apiKey = apiKey;
    }

    async generateText(model: string, system: string, messages: LLMMessage[], maxTokens?: number, temperature?: number): Promise<AdapterResponse> {
         return super.generateText(model, system, messages, maxTokens, temperature, {
            venice_parameters: { disable_thinking: true }
         });
    }

    async generateImage(options: ImageGenerationOptions): Promise<{ imageUrl: string; usage: AdapterUsage }> {
        const { model, prompt, width, height, referenceImage } = options;

        try {
            // Check if we should use image edit endpoint for inpainting
            if (referenceImage) {
                // Handle reference image array - Venice only accepts single image
                const imageArray = Array.isArray(referenceImage) ? referenceImage : [referenceImage];

                if (imageArray.length > 1) {
                    console.warn('Venice image edit only supports single image, using the first one');
                }

                // Venice edit endpoint only accepts single image, use the first one
                const imageData = imageArray[0];

                // Convert to proper base64 format
                let base64Image: string;
                if (imageData.startsWith('data:image')) {
                    // Already has data URL prefix, extract base64 part
                    base64Image = imageData.split(',')[1];
                } else {
                    // Raw base64
                    base64Image = imageData;
                }

                const requestBody: VeniceImageEditRequest = {
                    prompt,
                    image: base64Image
                };

                const response = await fetch(`${this.veniceBaseUrl}/image/edit`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.apiKey}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) {
                    const error = await response.text().catch(() => response.statusText);
                    throw new LLMError(
                        error || `Venice image edit error: ${response.status}`,
                        'venice',
                        undefined,
                        response.status
                    );
                }

                // Venice /image/edit returns PNG image directly, not JSON
                const imageBuffer = await response.arrayBuffer();
                const base64 = Buffer.from(imageBuffer).toString('base64');
                const imageUrl = `data:image/png;base64,${base64}`;

                const usage: AdapterUsage = {
                    itoken: 0,
                    otoken: 0,
                    ttoken: 0
                };

                return {
                    imageUrl,
                    usage
                };
            }

            // Regular image generation (no reference image)
            const requestBody: VeniceImageRequest = {
                model: model || 'lustify-sdxl',
                prompt,
                variants: 1,
                safe_mode: false,
                format: 'png'
            };

            // Set dimensions if provided
            if (width) {
                requestBody.width = Math.min(width, 1280);
            }
            if (height) {
                requestBody.height = Math.min(height, 1280);
            }

            const response = await fetch(`${this.veniceBaseUrl}/image/generate`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${this.apiKey}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({ error: response.statusText })) as any;
                throw new LLMError(
                    error.error || `Venice image generation error: ${response.status}`,
                    'venice',
                    undefined,
                    response.status
                );
            }

            const data: VeniceImageResponse = await response.json() as any;

            if (!data.images || data.images.length === 0) {
                throw new LLMError('No images generated', 'venice');
            }

            // Venice returns base64 encoded images
            const imageUrl = data.images[0].startsWith('data:')
                ? data.images[0]
                : `data:image/png;base64,${data.images[0]}`;

            const usage: AdapterUsage = {
                itoken: 0,
                otoken: 0,
                ttoken: 0
            };

            return {
                imageUrl,
                usage
            };
        } catch (error: any) {
            if (error instanceof LLMError) {
                throw error;
            }
            throw new LLMError(
                error.message || 'Venice image generation error',
                'venice',
                error.code
            );
        }
    }
}