// src/hooks/useScenario.ts
import { useState, useEffect } from 'react';

interface Translation {
    source_lang: string;
    source_label: string;
    target_lang: string;
    target_label: string;
    original: string;
    translation: string;
    translation_time: number;
    model: string;
    calver: string;
    id: string;
}

export interface Scenario {
    id: string;
    filename: string;
    source_lang: string;
    source_label: string;
    target_lang: string;
    target_label: string;
}

interface Manifest {
    scenarios: Scenario[];
}

interface UseScenarioReturn {
    scenario: Scenario | null;
    translations: Translation[];
    loading: boolean;
    error: string | null;
}

const GITHUB_ENDPOINT = 'https://raw.githubusercontent.com/ryderwishart/swarm/refs/heads/master/swarm_translate/scenarios/consolidated';
const DEV_ENDPOINT = '/consolidated';

export function useScenario(id: string | undefined): UseScenarioReturn {
    const [scenario, setScenario] = useState<Scenario | null>(null);
    const [translations, setTranslations] = useState<Translation[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const loadScenarioAndTranslations = async () => {
            if (!id) {
                setLoading(false);
                return;
            }

            try {
                // First, load the manifest to get scenario data
                const manifestResponse = await fetch('/manifest.json');
                if (!manifestResponse.ok) {
                    throw new Error('Failed to fetch manifest');
                }
                const manifestData: Manifest = await manifestResponse.json();
                const matchingScenario = manifestData.scenarios.find(s => s.id === id);

                if (!matchingScenario) {
                    throw new Error('Translation project not found');
                }

                setScenario(matchingScenario);

                // Determine if this is a Luke translation (local) or other translation (remote)
                const isLukeTranslation = id.includes('_luke');

                // Choose the appropriate endpoint based on translation type and environment
                let endpoint;
                if (isLukeTranslation) {
                    // Luke translations are always in the public directory
                    endpoint = '';
                } else {
                    // For regular translations, use GitHub in production, local in development
                    endpoint = process.env.NODE_ENV === 'production' ? GITHUB_ENDPOINT : DEV_ENDPOINT;
                }

                // Then, load the translations
                const translationsResponse = await fetch(`${endpoint}/${matchingScenario.filename}`);
                if (!translationsResponse.ok) {
                    throw new Error('Failed to fetch translations');
                }

                const text = await translationsResponse.text();
                console.log('Raw translation file content:', text.slice(0, 500)); // Log the first 500 chars

                const lines = text.split('\n');
                console.log('First few lines:', lines.slice(0, 3)); // Log first 3 lines

                const parsedTranslations: Translation[] = [];

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        console.log('Attempting to parse line:', line); // Log each line before parsing
                        const translation = JSON.parse(line) as Translation;
                        parsedTranslations.push(translation);
                    } catch (err) {
                        console.error('Error parsing line:', {
                            line: line,
                            length: line.length,
                            firstFewChars: line.slice(0, 20),
                            charCodes: Array.from(line.slice(0, 5)).map(c => c.charCodeAt(0))
                        });
                    }
                }

                setTranslations(parsedTranslations);
                setLoading(false);

            } catch (err) {
                setError(err instanceof Error ? err.message : 'An error occurred');
                setLoading(false);
            }
        };

        setLoading(true);
        setError(null);
        loadScenarioAndTranslations();
    }, [id]);

    return { scenario, translations, loading, error };
}