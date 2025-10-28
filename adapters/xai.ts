import { OpenAIAdapter } from './openai';

export class XAIAdapter extends OpenAIAdapter {
    constructor(apiKey: string) {
        super(apiKey, 'https://api.x.ai/v1');
    }
}