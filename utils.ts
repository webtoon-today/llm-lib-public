import { LLMProvider, LLMError } from './index';

const randomString = (length: number) =>
    Array(length).fill(0).map(() => `abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`.charAt(Math.floor(Math.random() * 62))).join('');

export function generateTrackId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c == 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
    });
}

export function log(level: 'info' | 'warn' | 'error', errorLevel: 'quiet' | 'info' | 'warn' | 'error' = 'quiet', message: string, ...args: any[]) {
    const levels = { quiet: 0, info: 1, warn: 2, error: 3 };
    const currentLevel = levels[errorLevel];
    const messageLevel = levels[level];
    
    if (messageLevel >= currentLevel) {
        switch (level) {
            case 'info':
                console.log(`[LLM-INFO] ${message}`, ...args);
                break;
            case 'warn':
                console.warn(`[LLM-WARN] ${message}`, ...args);
                break;
            case 'error':
                console.error(`[LLM-ERROR] ${message}`, ...args);
                break;
        }
    }
}

export async function logTracking(tracking: any) {
    const logData = {
        timestamp: new Date().toISOString(),
        level: 'INFO',
        logger: 'LLM_TRACKING',
        ...tracking
    };

    // Log to console
    console.log(JSON.stringify(logData));
}

export async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

export async function retryWithBackoff<T>(
    fn: () => Promise<T>,
    retries: number,
    initialDelay: number = 1000
): Promise<T> {
    let lastError: any;
    
    for (let i = 0; i <= retries; i++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (i < retries) {
                const delay = initialDelay * Math.pow(2, i);
                await sleep(delay);
            }
        }
    }
    
    throw lastError;
}

// Cache for API keys
const apiKeyCache: { [key: string]: string } = {};

export async function getApiKey(provider: LLMProvider): Promise<string> {
    // Kling is handled differently (uses its own adapter)
    if (provider === 'kling') {
        return 'HANDLED_BY_ADAPTER';
    }

    const envVarNames = {
        anthropic: 'ANTHROPIC_API_KEY',
        google: 'GOOGLE_AI_API_KEY',
        openai: 'OPENAI_API_KEY',
        venice: 'VENICE_API_KEY',
        xai: 'XAI_API_KEY'
    };

    const envVarName = envVarNames[provider as keyof typeof envVarNames];
    if (!envVarName) {
        throw new LLMError(`Unknown provider: ${provider}`, provider, 'UNKNOWN_PROVIDER');
    }

    // Check cache first
    if (apiKeyCache[envVarName]) {
        return apiKeyCache[envVarName];
    }

    // Get from environment variable (optional)
    const apiKey = process.env[envVarName];

    if (!apiKey) {
        throw new LLMError(
            `API key not found for ${provider}. Please set ${envVarName} environment variable.`,
            provider,
            'MISSING_API_KEY'
        );
    }

    // Cache the key
    apiKeyCache[envVarName] = apiKey;

    return apiKey;
}

// Parse JSON from LLM output - handles common formatting issues
export function parseLLMJson<T = any>(jsonString: string): T {
    try {
        // Step 1: Remove all actual newlines (preserves escaped \n)
        let cleaned = jsonString.replace(/\n/g, '');
        
        // Step 2: Handle common LLM formatting issues
        // Remove markdown code blocks if present
        cleaned = cleaned.replace(/^```json\s*/, '').replace(/\s*```$/, '');
        cleaned = cleaned.replace(/^```\s*/, '').replace(/\s*```$/, '');
        
        // Step 3: Trim whitespace
        cleaned = cleaned.trim();
        
        // Step 4: Parse the cleaned JSON
        return JSON.parse(cleaned);
    } catch (error) {
        // Step 5: If parsing still fails, try to extract JSON from the string
        const jsonMatch = jsonString.match(/\{[\s\S]*\}|\[[\s\S]*\]/);
        if (jsonMatch) {
            try {
                const extracted = jsonMatch[0];
                // Recursively clean the extracted JSON
                return parseLLMJson(extracted);
            } catch (innerError) {
                throw new Error(`Failed to parse LLM JSON: ${error}`);
            }
        }
        
        throw new Error(`Failed to parse LLM JSON: ${error}`);
    }
}

// Validate base64 string
export function isValidBase64(str: string): boolean {
    if (!str || typeof str !== 'string') {
        return false;
    }
    
    try {
        // Remove data URL prefix if present
        const base64String = str.replace(/^data:image\/[a-zA-Z]+;base64,/, '');
        
        // Check if string contains valid base64 characters
        const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
        if (!base64Regex.test(base64String)) {
            return false;
        }
        
        // Check if string length is valid (must be multiple of 4)
        if (base64String.length % 4 !== 0) {
            return false;
        }
        
        // Try to decode - if it fails, it's invalid
        Buffer.from(base64String, 'base64');
        return true;
    } catch (error) {
        return false;
    }
}

// Return base64 image data URL (no cloud storage upload)
export async function uploadBase64ImageToS3(base64Data: string, prefix: string = 'llm-generated'): Promise<string | null> {
    try {
        // Simply return the base64 data URL as-is
        // If it doesn't have a data URL prefix, add one
        if (!base64Data.startsWith('data:')) {
            return `data:image/png;base64,${base64Data}`;
        }
        return base64Data;
    } catch (error) {
        log('error', 'error', 'Error processing image:', error);
        return null;
    }
}