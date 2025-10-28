export interface ImageGenerationOptions {
    model: string;
    prompt: string;
    system?: string;
    width?: number;
    height?: number;
    referenceImage?: string | string[];
}

export type AdapterUsage = {
    itoken: number;
    otoken: number;
    ttoken: number;
};

export type AdapterResponse = {
    response: string;
    usage: AdapterUsage;
};

export type StreamChunk = {
    type: 'text' | 'usage';
    text?: string;
    usage?: AdapterUsage;
};