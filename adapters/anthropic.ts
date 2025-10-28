import Anthropic from '@anthropic-ai/sdk';
import { LLMMessage, LLMError } from '../index';
import { AdapterResponse, StreamChunk } from '../types';

export class AnthropicAdapter {
    private client: Anthropic;

    constructor(apiKey: string) {
        this.client = new Anthropic({ apiKey });
    }

    async generateText(
        model: string,
        system: string,
        messages: LLMMessage[],
        maxTokens: number = 4096,
        temperature: number = 0.7
    ): Promise<AdapterResponse> {
        try {
            const anthropicMessages = this.convertMessages(messages);
            
            const response = await this.client.messages.create({
                model,
                system,
                messages: anthropicMessages,
                max_tokens: maxTokens,
                temperature
            });

            const text = response.content[0].type === 'text' 
                ? response.content[0].text 
                : '';

            return {
                response: text,
                usage: {
                    itoken: response.usage.input_tokens,
                    otoken: response.usage.output_tokens,
                    ttoken: 0 // Anthropic does not provide distinguished thinking tokens(included in output tokens)
                }
            };
        } catch (error: any) {
            throw new LLMError(
                error.message || 'Anthropic API error',
                'anthropic',
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
            const anthropicMessages = this.convertMessages(messages);
            
            const stream = await this.client.messages.create({
                model,
                system,
                messages: anthropicMessages,
                max_tokens: maxTokens,
                temperature,
                stream: true
            });

            let streamUsage: { itoken: number; otoken: number; ttoken: number } | undefined;

            for await (const chunk of stream) {
                // Handle different chunk types from Anthropic streaming
                if (chunk.type === 'message_start') {
                    // Initial message with input token count
                    streamUsage = {
                        itoken: chunk.message.usage.input_tokens,
                        otoken: 0, // Will be updated later
                        ttoken: 0
                    };
                } else if (chunk.type === 'message_delta') {
                    // Message delta may contain final usage info
                    if (chunk.usage && streamUsage) {
                        streamUsage.otoken = chunk.usage.output_tokens;
                    }
                } else if (chunk.type === 'content_block_delta' && 
                    chunk.delta.type === 'text_delta') {
                    yield {
                        type: 'text',
                        text: chunk.delta.text
                    };
                } else if (chunk.type === 'message_stop') {
                    // Stream completed - yield final usage data
                    if (streamUsage) {
                        yield {
                            type: 'usage',
                            usage: streamUsage
                        };
                    }
                    break;
                }
            }
        } catch (error: any) {
            throw new LLMError(
                error.message || 'Anthropic API error',
                'anthropic',
                error.code,
                error.status
            );
        }
    }

    private convertMessages(messages: LLMMessage[]): Anthropic.MessageParam[] {
        return messages
            .filter(msg => msg.role !== 'system')
            .map(msg => ({
                role: msg.role as 'user' | 'assistant',
                content: typeof msg.content === 'string' 
                    ? msg.content 
                    : msg.content.map(c => {
                        if (c.type === 'text') {
                            return { type: 'text', text: c.text || '' };
                        } else if (c.type === 'image') {
                            return { 
                                type: 'image', 
                                source: {
                                    type: 'base64',
                                    media_type: 'image/png',
                                    data: c.image || ''
                                }
                            };
                        }
                        return { type: 'text', text: '' };
                    })
            }));
    }
}