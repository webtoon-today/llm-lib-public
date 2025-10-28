import axios from 'axios';
import * as jwt from 'jsonwebtoken';
import { LLMMessage, LLMError } from '../index';
import { ImageGenerationOptions, AdapterResponse } from '../types';

type KlingCredential = {
    id: string;
    secret: string;
}

type KlingImageGenerationResponse = {
    code: number;
    message: string;
    data?: {
        task_id: string;
        task_status: string;
        created_at: number;
    };
}

type KlingTaskStatusResponse = {
    code: number;
    message: string;
    data?: {
        task_id: string;
        task_status: 'submitted' | 'processing' | 'succeed' | 'failed';
        task_result?: {
            images: Array<{
                url: string;
            }>;
        };
        created_at: number;
        updated_at: number;
    };
}

export class KlingAdapter {
    private credential: KlingCredential | null = null;

    constructor() {}

    private async getCredential(): Promise<KlingCredential> {
        if (!this.credential) {
            const klingId = process.env.KLING_ACCESS_KEY_ID;
            const klingSecret = process.env.KLING_ACCESS_KEY_SECRET;

            if (!klingId || !klingSecret) {
                throw new LLMError(
                    'Kling credentials not found. Please set KLING_ACCESS_KEY_ID and KLING_ACCESS_KEY_SECRET environment variables.',
                    'kling',
                    'MISSING_CREDENTIALS'
                );
            }

            this.credential = {
                id: klingId,
                secret: klingSecret
            };
        }

        return this.credential;
    }

    private async generateJWT(): Promise<string> {
        const cred = await this.getCredential();
        
        return jwt.sign({
            "iss": cred.id,
            "exp": Math.floor(Date.now() / 1000) + 60 * 30, // 30 minutes
            "nbf": Math.floor(Date.now() / 1000) - 5,
        }, cred.secret, { header: { alg: 'HS256', typ: 'JWT' } });
    }

    async generateText(
        model: string,
        system: string,
        messages: LLMMessage[],
        maxTokens: number = 4096,
        temperature: number = 0.7
    ): Promise<AdapterResponse> {
        throw new LLMError('Text generation not supported by Kling', 'kling', 'NOT_SUPPORTED');
    }

    async *generateStream(
        model: string,
        system: string,
        messages: LLMMessage[],
        maxTokens: number = 4096,
        temperature: number = 0.7
    ): AsyncGenerator<string> {
        throw new LLMError('Streaming not supported by Kling', 'kling', 'NOT_SUPPORTED');
    }

    async generateImage(options: ImageGenerationOptions): Promise<{ imageUrl: string; usage: { itoken: number; otoken: number; ttoken: number } }> {
        const { model, prompt, system, width, height, referenceImage } = options;
        try {
            const token = await this.generateJWT();
            
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
            
            // Prepare request body
            const requestBody: any = {
                model_name: model || 'kling-v1-5',
                prompt: system ? `${system}\n\nDescription: ${prompt}` : prompt,
                n: 1,
                aspect_ratio: aspectRatio,
            };

            // Add reference image if provided
            if (referenceImage) {

                let imageData = Array.isArray(referenceImage) ? referenceImage[0] : referenceImage;
                // Extract base64 data if it's a data URL
                if (imageData.startsWith('data:image')) {
                    imageData = imageData.split(',')[1];
                }
                requestBody.image = imageData;
                requestBody.image_reference = 'face';
                requestBody.image_fidelity = 0.3;
                requestBody.human_fidelity = 0.9;
            }

            // Make request to Kling AI
            const response = await axios.post<KlingImageGenerationResponse>(
                'https://api-singapore.klingai.com/v1/images/generations',
                requestBody,
                {
                    headers: {
                        Authorization: `Bearer ${token}`,
                        'Content-Type': 'application/json',
                    }
                }
            );

            if (response.status !== 200 || response.data.code !== 0) {
                throw new LLMError(
                    response.data.message || 'Failed to generate image',
                    'kling',
                    String(response.data.code),
                    response.status
                );
            }

            const { task_id, task_status } = response.data.data || {};

            if (!task_id || !task_status) {
                throw new LLMError('Missing task data in response', 'kling', 'INVALID_RESPONSE');
            }

            // Poll for completion
            const imageUrl = await this.pollForCompletion(task_id, token);
            
            // Kling returns URLs, not base64
            return {
                imageUrl,
                usage: { itoken: 0, otoken: 0, ttoken: 0 } // Kling doesn't provide token usage information
            };

        } catch (error: any) {
            if (error instanceof LLMError) {
                throw error;
            }
            throw new LLMError(
                error.message || 'Kling API error',
                'kling',
                error.code,
                error.response?.status
            );
        }
    }

    private async pollForCompletion(taskId: string, token: string): Promise<string> {
        const maxAttempts = 60; // 10 minutes with 10 second intervals
        const pollInterval = 10000; // 10 seconds

        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            const response = await axios.get<KlingTaskStatusResponse>(
                `https://api-singapore.klingai.com/v1/images/get-result?task_id=${taskId}`,
                {
                    headers: {
                        Authorization: `Bearer ${token}`,
                    }
                }
            );

            if (response.status !== 200 || response.data.code !== 0) {
                throw new LLMError(
                    response.data.message || 'Failed to get task status',
                    'kling',
                    String(response.data.code),
                    response.status
                );
            }

            const { task_status, task_result } = response.data.data || {};

            if (task_status === 'succeed' && task_result?.images?.[0]?.url) {
                const imageUrl = task_result.images[0].url;
                
                // Download the image and convert to base64
                try {
                    const imageResponse = await axios.get(imageUrl, {
                        responseType: 'arraybuffer',
                        timeout: 30000 // 30 seconds timeout
                    });
                    
                    const base64 = Buffer.from(imageResponse.data).toString('base64');
                    const contentType = imageResponse.headers['content-type'] || 'image/png';
                    
                    // Return as data URL for S3 upload in index.ts
                    return `data:${contentType};base64,${base64}`;
                } catch (downloadError) {
                    // If download fails, throw error
                    throw new LLMError('Failed to download image from Kling', 'kling', 'DOWNLOAD_FAILED');
                }
            } else if (task_status === 'failed') {
                throw new LLMError('Image generation failed', 'kling', 'GENERATION_FAILED');
            }

            // Wait before next poll
            await new Promise(resolve => setTimeout(resolve, pollInterval));
        }

        throw new LLMError('Image generation timed out', 'kling', 'TIMEOUT');
    }
}