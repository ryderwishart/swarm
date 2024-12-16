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

const ENDPOINT = (() => {
    const endpoint = process.env.NODE_ENV === 'production'
        ? 'https://raw.githubusercontent.com/ryderwishart/swarm/refs/heads/master/swarm_translate/scenarios/consolidated'
        : 'http://localhost:5173';
    return endpoint;
})();

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

                // Then, load the translations
                const translationsResponse = await fetch(`${ENDPOINT}/${matchingScenario.filename}`);
                if (!translationsResponse.ok) {
                    throw new Error('Failed to fetch translations');
                }

                const text = await translationsResponse.text();
                const lines = text.split('\n');
                const parsedTranslations: Translation[] = [];

                for (const line of lines) {
                    if (!line.trim()) continue;
                    try {
                        const translation = JSON.parse(line) as Translation;
                        parsedTranslations.push(translation);
                    } catch (err) {
                        console.error('Error parsing translation line:', err);
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