// ==UserScript==
// @name         Webpage Text Translator
// @author       Qing Wen
// @namespace    http://tampermonkey.net/
// @version      0.3
// @description  Extract and translate text from webpage
// @match        https://*.syosetu.com/*/*/
// @match        https://kakuyomu.jp/works/*/episodes/*
// @grant        GM_setValue
// @grant        GM_getValue
// ==/UserScript==

'use strict';

const STORAGE_KEYS = {
    OPENROUTER: 'openrouter_api_key',
    ANTHROPIC: 'anthropic_api_key',
    OPENAI: 'openai_api_key',
    PROVIDER: 'selected_provider'
};

const API_ENDPOINTS = {
    OPENROUTER: 'https://openrouter.ai/api/v1/chat/completions',
    ANTHROPIC: 'https://api.anthropic.com/v1/messages',
    OPENAI: 'https://api.openai.com/v1/chat/completions'
};

const PROVIDER_CONFIGS = {
    OPENROUTER : {
        id: 'openrouter',
        label: 'OpenRouter',
        S1A_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'google/gemini-pro-1.5', 'providers':['Google AI Studio', 'Google']}, label: 'Gemini Pro 1.5' },
            { value: {'model': 'anthropic/claude-3.5-sonnet:beta', 'providers':['Anthropic']}, label: 'Sonnet 3.5' },
            { value: {'model': 'openai/gpt-4o-2024-11-20', 'providers':['OpenAI']}, label: 'GPT-4o' }
        ],
        S1B_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'qwen/qwen-2.5-coder-32b-instruct', 'providers':['Lambda', 'DeepInfra', 'Hyperbolic']}, label: 'Qwen2.5 Coder' },
            { value: {'model': 'amazon/nova-micro-v1', 'providers':['Amazon Bedrock']}, label: 'Nova Micro 1.0' },
            { value: {'model': 'meta-llama/llama-3.3-70b-instruct', 'providers':['Lambda', 'Nebius']}, label: 'Llama 3.3 70B' },
            { value: {'model': 'mistralai/mistral-small-24b-instruct-2501', 'providers':['DeepInfra', 'Mistral']}, label: 'Mistral Small 3' },
            { value: {'model': 'microsoft/phi-4', 'providers':['DeepInfra']}, label: 'Phi 4' },
            { value: {'model': 'qwen/qwen-turbo', 'providers':['Alibaba']}, label: 'Qwen Turbo' },
            { value: {'model': 'google/gemini-2.0-flash-001', 'providers':['Google AI Studio']}, label: 'Gemini 2.0 Flash' }
        ],
        S2_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'deepseek/deepseek-chat', 'providers':['DeepInfra', 'Nebius', 'Fireworks', 'Novita', 'DeepSeek']}, label: 'DeepSeek V3' },
            { value: {'model': 'google/gemini-2.0-flash-001', 'providers':['Google AI Studio']}, label: 'Gemini Flash 2.0' },
            { value: {'model': 'openai/gpt-4o-mini', 'providers':['OpenAI']}, label: 'GPT-4o Mini' },
            { value: {'model': 'qwen/qwen-2.5-72b-instruct', 'providers':['DeepInfra', 'Novita', 'Hyperbolic']}, label: 'Qwen2.5 72B' },
            { value: {'model': 'qwen/qwen-plus', 'providers':['Alibaba']}, label: 'Qwen Plus' }
        ]
    },
    ANTHROPIC: {
        id: 'anthropic',
        label: 'Anthropic',
        S1A_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'claude-3-5-sonnet-latest'}, label: 'Sonnet 3.5' }
        ],
        S1B_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'claude-3-haiku-20240307'}, label: 'Haiku 3' }
        ],
        S2_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'claude-3-haiku-20240307'}, label: 'Haiku 3' }
        ]
    },
    OPENAI: {
        id: 'openai',
        label: 'OpenAI',
        S1A_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'gpt-4o'}, label: 'GPT-4o' }
        ],
        S1B_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'gpt-4o-mini'}, label: 'GPT-4o-mini' }
        ],
        S2_OPTIONS: [
            { value: '', label: 'Select Model' },
            { value: {'model': 'gpt-4o-mini'}, label: 'GPT-4o-mini' }
        ]
    }
};



// ================================================================================
// Globla variables
// ================================================================================

// Currently selected models
let models = {
    "stage1a": {"model": "", "providers": ""},
    "stage1b":{"model": "", "providers": ""},
    "stage2":{"model": "", "providers": ""}
};

// User selected values
let selectedProvider;
let currentProvider;

// UI elements
let errorCount;
let errorDisplay;
let progressDisplay;
let floatingBox;

// Saves a old state of dictionary, allow for undo in various functions
let dictionary;

// Domain specific config, set on page load
let domainManager;

// Loads API keys once on page load
let OPENROUTER_API_KEY;
let ANTHROPIC_API_KEY;
let OPENAI_API_KEY;

// ================================================================================
// Stage 1a - Generate dictionary
// ================================================================================

async function generateDictionary(texts) {
    const systemPrompt = `You are generating entries to a multi-key dictionary that will be used as the knowledge base for a RAG pipeline in a Japanese to English LLM translation task. This metadata part will be used to help ensure consistency in translation of names and proper nouns across multiple API calls. In the RAG pipeline, the text to be translated will be scanned and presence of any of keys will result in the content of "value" to be included in the LLM context.

Rules and Guidelines:
1. Output Format:
   - Generate a JSON structure with an "entries" array
   - Each entry should contain "keys" (array of Japanese terms/names) and "value" (structured metadata string)

2. Value String Format:
   - Start with category in square brackets (e.g., [character], [location], [organization], [term], [skill name])
   - Use "Key: Value" pairs separated by " | " (space, pipe, space)
   - For names, write the full English name, followed by the full Japanese versions in brackets when applicable
   - Always include gender information for characters unless undeterminable
   - Use appropriate capitalization for proper nouns
   - Be concise but include relevant details

3. Entry Selection Criteria:
   - Focus on character names, location names, proper nouns, and special terms
   - Exclude common nouns or terms
   - Skip terms if unclear or lacking sufficient context
   - Include variations of Japanese terms that might appear in the text

4. LLM Compatibility:
   - Keep total length concise to efficiently use context space
   - Ensure additional information is directly relevant to translation
   - Include helpful information like nicknames when beneficial
   - Maintain consistent formatting for reliable parsing

Expected JSON Structure:
{
  "entries": [
    {
      "keys": [array of strings],
      "value": "[category] Key: Value | Additional_Field: Additional_Value | ..."
    }
  ]
}

Example Output:
{
  "entries": [
    {
      "keys": ["名無しの権兵衛", "ななしのごんべい"],
      "value": "[character] Name: John Doe (名無しの権兵衛) | Gender: Male | Nickname: Nanashi (ななし) | Notes: Brief description or explanation ..."
    },
    {
      "keys": ["アメリカ合衆国", "アメリカ"],
      "value": "[location] Name: United States (アメリカ) | Notes: Lorem ipsum dolor sit amet ..."
    }
  ]
}

You will be provided the raw text delimited with <text> XML tags.`;

    const sentences = Array.from(texts.values());
    const chunks = [];
    let currentChunk = [];
    let currentTotal = 0;

    // Split sentences into chunks <= 1000 characters
    for (const sentence of sentences) {
        const len = sentence.length;
        if (currentTotal + len > 5000) {
            if (currentChunk.length > 0) {
                chunks.push(currentChunk);
                currentChunk = [];
                currentTotal = 0;
            }
            currentChunk.push(sentence);
            currentTotal = len;
        } else {
            currentChunk.push(sentence);
            currentTotal += len;
        }
    }
    if (currentChunk.length > 0) chunks.push(currentChunk);

    // Process chunks in parallel
    const translationQueue = new TranslationQueue();
    translationQueue.setTextDisplay('Generating glossary ...');
    translationQueue.setProgressDisplay(progressDisplay, chunks.length);

    const chunkPromises = chunks.map(chunk =>
                                     translationQueue.enqueueTask(async () => {
        const userPrompt = `<text>\n${chunk}\n</text>`;
        const prompt = {
            system: systemPrompt,
            user: userPrompt,
            assistant: `{"entries": [`
        };
        const reply = await llm_api(prompt, models.stage1a, 4096);
        const jsonData = extractJsonFromLLMOutput(reply);
        return jsonData?.entries || [];
    }));

    // Wait for all chunk processing to complete
    const results = await Promise.allSettled(chunkPromises);

    // Aggregate results
    const allEntries = [];
    for (const result of results) {
        if (result.status === "fulfilled") {
            allEntries.push(...result.value);
        }
    }

    return { entries: allEntries };
}

// ================================================================================
// Stage 1b - Update / merge dictionary
// ================================================================================

async function updateDictionary(existingDict, newUpdates) {
    const systemPrompt = `You are a dictionary entry merger for a translation system using a RAG pipeline. Your task is to analyze existing entries and a new update, then output a merged dictionary that either updates existing entries or creates new ones with appropriate IDs while maintaining strict consistency in translations.
The dictionary you are working with is generated from a block of text from a web novel. The existing dictionary is created from all previous chapter, and the new updates are generated from the newest chapter.
During translation, only the 'value' will be inserted in the context, therefore it is important that it contains both the English and Japanese (in brackets) text. The keys are used by the RAG system to determine which entrries are inserted into the LLM context.

Operational parameters:
1. For matching entries:
 - Preserve existing IDs
 - Always retain original translations of names and terms
2. For novel entries:
 - Assign new IDs sequential to max_id
 - Create comprehensive key lists including all variants
3. Value string rules:
 - Preserve original translations even if alternates appear in new updates
 - The new updates are obtained from a later portion of the story, update the description as needed. You should only add relevant addtional information
 - If the there is conflict in the information provided, discard the new updates
 - Merge similar fields, by combining the information
4. Key management:
 - Existing keys are immutable and must be preserved exactly
 - Keys should only contain Japanese text, and never English strings

Output schema:
{
  "entries": [
    {
      "id": number,      // Preserved if updating existing entry, (max_id + 1) for creating new entry
      "keys": string[],  // Union of existing and new keys
      "value": string    // '[type] Field1: Value1 | Field2: Value2'
    }
  ]
}

Example input:
<existing_dictionary>
{
  "entries": [
    {
      "id": 7,
      "keys": ['あ'],
      "value": '[term] Lorem ipsum (いろはにほへと) | Notes: xxx'
    }
  ]
}
</existing_dictionary>
<new_updates>
{
  "entries": [
    {
      "keys": ['え'],
      "value": '[term] Lorem ipsum (いろはにほへと) | Notes: yyy'
    }
  ]
}
</new_updates>
Example output:
{
  "entries": [
    {
      "id": 7,
      "keys": ['あ', 'え'],
      "value": [term] Lorem ipsum (いろはにほへと) | Notes: xxx, yyy'
    }
  ]
}`;
    const updatedDict = structuredClone(existingDict);

    for (const newEntry of newUpdates.entries) {
        const relevantEntries = findRelevantEntries(updatedDict.entries, [newEntry]);

        // No conflict, therefore direct append
        if (!relevantEntries.length) {
            processDictionaryMutation(updatedDict, newEntry);
            continue;
        }

        // Solve conflict with LLM
        const maxId = Math.max(...existingDict.entries.map(entry => entry.id), 0);

        const prompt = {
            system: systemPrompt,
            user: constructUserPromptDictMerger(
                { entries: relevantEntries },
                { entries: [newEntry] },
                maxId
            ),
            assistant: `{"entries": [`
        };

        try {
            const llmOutput = await llm_api(prompt, models.stage1b, 2048);
            const mergedEntries = extractJsonFromLLMOutput(llmOutput);

            for (const mergedEntry of mergedEntries.entries) {
                const originalEntry = relevantEntries.find(e => e.id === mergedEntry.id);
                if (originalEntry) {
                    processDictionaryMutation(updatedDict, mergedEntry, originalEntry);
                } else {
                    processDictionaryMutation(updatedDict, newEntry);
                }
            }
        } catch (error) {
            console.error('Entry processing error:', error);
            continue;
        }
    }

    return updatedDict;
}


// Finds existing entries that share keys
function findRelevantEntries(existingEntries, newEntries) {
    const relevantIds = new Set();

    for (const newEntry of newEntries) {
        const newKeySet = new Set(newEntry.keys);

        for (const existingEntry of existingEntries) {
            // Check for any key overlap
            const hasOverlap = existingEntry.keys.some(key => newKeySet.has(key));

            if (hasOverlap) {
                relevantIds.add(existingEntry.id);
            }
        }
    }

    return existingEntries.filter(entry => relevantIds.has(entry.id));
}

function constructUserPromptDictMerger(existingDict, newUpdates, maxId) {
    return `<dictionary_metadata>
max_id: ${maxId}
</dictionary_metadata>
<existing_dictionary>
${JSON.stringify(existingDict, null, 2)}
</existing_dictionary>
<new_updates>
${JSON.stringify(newUpdates.entries, null, 2)}
</new_updates>`;
}

// Validates entry structure and semantics
function validateDictionaryEntry(entry, originalEntry = null, isMergeValidation = false) {
    // Schema validation
    const requiredKeys = ['keys', 'value'];
    const allowedKeys = [...requiredKeys, 'id'];

    const hasValidKeys = Object.keys(entry).every(key => allowedKeys.includes(key)) &&
          requiredKeys.every(key => key in entry);
    if (!hasValidKeys) {
        console.warn('Entry schema validation failed:', entry);
        return false;
    }

    // Type validation
    if (!Array.isArray(entry.keys) ||
        entry.keys.length === 0 ||
        !entry.keys.every(k => typeof k === 'string') ||
        typeof entry.value !== 'string') {
        console.warn('Entry type validation failed:', entry);
        return false;
    }

    // Merge-specific validation
    if (isMergeValidation && originalEntry) {
        // ID existence
        if (entry.id !== originalEntry.id) {
            console.warn('Merge ID mismatch:', entry.id, originalEntry.id);
            return false;
        }

        // Keys monotonicity
        const originalKeySet = new Set(originalEntry.keys);
        const hasValidKeyExpansion = originalEntry.keys.every(k => entry.keys.includes(k));
        if (!hasValidKeyExpansion) {
            console.warn('Invalid key modification detected:',
                         'Original:', originalEntry.keys,
                         'New:', entry.keys);
            return false;
        }
    }

    return true;
}

// Processes dictionary mutations with validation
function processDictionaryMutation(updatedDict, entry, originalEntry = null) {
    const isMergeOperation = originalEntry !== null;

    if (!validateDictionaryEntry(entry, originalEntry, isMergeOperation)) {
        return false;
    }

    if (isMergeOperation) {
        const existingIndex = updatedDict.entries.findIndex(e => e.id === entry.id);
        if (existingIndex !== -1) {
            updatedDict.entries[existingIndex] = entry;
            return true;
        }
        return false;
    } else {
        const maxId = Math.max(...updatedDict.entries.map(e => e.id), 0);
        updatedDict.entries.push({
            ...entry,
            id: maxId + 1
        });
        return true;
    }
}

// ================================================================================
// Stage 2 - Translation
// ================================================================================

async function translateLine(previousLines, currentLine, dictionary) {
    const systemPrompt = `You are a professional translator specializing in translating Japanese novels into English. Your task is to accurately translate the given text while maintaining the original style, tone, and cultural nuances. Metadata and additional instructions (delimited with XML tags) will be provided, which you should use to guide your translation.
If there are any conflicts between the metadata and the actual text being translated, always treat the raw text as the source of truth and ignore conflicting metadata information. Use the context provided by previous lines of text to ensure the translation is coherent and consistent with the narrative.

Before you start translating, think through step by step using <thinking> and </thinking> tags:
- Analyze the sentence structure and identify any grammatical patterns or idiomatic expressions
- Consider the context provided by previous lines and metadata to understand the overall meaning
- Determine the appropriate English tense, voice, and sentence structure to convey the original meaning
- Check for any names, proper nouns or terms mentioned in the metadata and use them if provided and is relevant
- The proper pronouns (if necessary) to be used in the English translation, and if they are not present in the original Japanese text, use the additional context or metadata provided to determine this information
- Check for additional instruction the user provided, and follow them
Once you are prepared and ready, write the translated sentence inside <output> and </output> tags. Remember to properly close your tags with '</'!`;

    const userPrompt = constructUserPromptTranslation(previousLines, currentLine, dictionary);

    const prompt = {'system' : systemPrompt,
                    'user': userPrompt,
                    'assistant': '<thinking>'}

    const reply = await llm_api(prompt, models.stage2, 2048);

    const translatedText = extractTextFromTag(reply, 'output');

    return translatedText;
}

function constructUserPromptTranslation(previousLines, currentLine, dictionary) {
    const metadata = generateMetadata([...previousLines, currentLine].join(' '), dictionary) + '\n';

    // Empty if nothing to add (the very first sentence)
    const preceedingText = previousLines
    ? `The following is a few lines of text that comes right BEFORE the sentence you are asked to translate. You should use them to help determine the context of the sentence you are asked to translate.
${previousLines.join('\n')}
`
    : '';

    return `<instructions>
When translating names, preserve honorifics if they are present in the original text. In addition, use the same first name / last name ordering in the translated text.
<example>
'花子さん' -> 'Hanako-san'
'山田太郎' -> 'Yamada Taro'
'花子様' -> 'Hanako-sama'
</example>
Text inside '「' and '」' are dialouge, and should be indicated as such in the transalted text with qdouble qoutes '"'.
The text you are translating is extracted form a HTML webpage, and text inside brackets '(' and ')' could be originally ruby annotations of the preceding few characters.
</instructions>
<metadata>${metadata}${preceedingText}</metadata>
Translate only the following sentence(s) from Japanese into English:
${currentLine}`;
}

// Returns all entries in the multi-key dictionary that matches the text.
function generateMetadata(text, dictionary) {
    const metadataArray = [];

    for (const entry of dictionary.entries) {
        if (entry.keys.some(key => text.includes(key))) {
            metadataArray.push(entry.value);
        }
    }

    if (metadataArray.length !== 0) {
        return '\n' + metadataArray.join('\n') + '\n';
    } else {
        return '';
    }
}

// ================================================================================
// LLM API
// ================================================================================

async function llm_api(prompt, model, max_tokens = 256) {
    try {
        const messages = [
            {
                role: 'system',
                content: prompt.system
            },
            {
                role: 'user',
                content: prompt.user
            }
        ];

        // Add prefill if exists
        if ('assistant' in prompt) {
            messages.push({
                role: 'assistant',
                content: prompt.assistant
            });
        }

        const endpoint = API_ENDPOINTS[currentProvider];
        const headers = {
            'Content-Type': 'application/json'
        };

        let payload;

        switch (currentProvider) {
            case 'OPENROUTER':
                headers.Authorization = `Bearer ${OPENROUTER_API_KEY}`;
                payload = {
                    model: model.model,
                    messages,
                    temperature: 0,
                    max_tokens,
                    provider: {
                        order: model.providers,
                        allow_fallbacks: false
                    }
                };
                break;

            case 'ANTHROPIC':
                headers['x-api-key'] = ANTHROPIC_API_KEY;
                headers['anthropic-version'] = '2024-01-01';
                payload = {
                    model: model.model,
                    messages,
                    max_tokens,
                    temperature: 0
                };
                break;

            case 'OPENAI':
                headers.Authorization = `Bearer ${OPENAI_API_KEY}`;
                payload = {
                    model: model.model,
                    messages,
                    max_tokens,
                    temperature: 0
                };
                break;

            default:
                throw new Error(`Unsupported provider: ${currentProvider}`);
        }

        const response = await fetch(endpoint, {
            method: 'POST',
            headers,
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        let completion;

        // Provider-specific response parsing
        switch (currentProvider) {
            case 'ANTHROPIC':
                completion = data.content[0].text;
                break;
            case 'OPENROUTER':
            case 'OPENAI':
                completion = data.choices[0].message.content;
                break;
            default:
                throw new Error(`Invalid provider: ${currentProvider}`);
        }

        console.log('[System]:\n' + prompt.system + '\n' + '-'.repeat(80) + '\n' +
                    '[User]:\n'+ prompt.user + '\n' + '-'.repeat(80) + '\n' +
                    '[Assistant]:\n'+ completion + '\n' + '='.repeat(80) + '\n');


        return completion;
    } catch (error) {
        console.error('Error in LLM API call:', error);
        console.error('Not processed: ' + prompt.user);
        errorHandler();
        return '### Error in LLM API call ###';
    }
}

function errorHandler() {
    errorCount++;
    if (errorCount > 0) {
        errorDisplay.textContent = `Errors: ${errorCount}`;
        errorDisplay.style.display = 'block';
    } else {
        errorDisplay.style.display = 'none';
    }
}

// ================================================================================
// Helper functions
// ================================================================================

function loadDictionary() {
    const seriesId = domainManager.getCurrentSeriesId();

    if (seriesId) {
        const dictionaryKey = domainManager.getDictionaryKey(seriesId);
        return GM_getValue(dictionaryKey, { entries: [] });
    } else {
        console.error('Failed to extract series ID from URL');
        return { entries: [] };
    }
}

function saveDictionary(dictionary) {
    const seriesId = domainManager.getCurrentSeriesId();

    if (seriesId) {
        const dictionaryKey = domainManager.getDictionaryKey(seriesId);
        GM_setValue(dictionaryKey, dictionary);
    } else {
        console.error('Failed to extract series ID from URL');
    }
}

// For dictionary processing
function extractJsonFromLLMOutput(output, prefill = '{"entries": [') {
    // Case: Output strucrture correct
    try {
        return JSON.parse(output);
    } catch (error) {
        console.warn('Unable to parse JSON directly, attempting fo fix errors. Original output\n: ' + output);
    }

    // Case: Prefill not included in response
    let trimmedOutput = output.trimStart();
    if (!trimmedOutput.startsWith(prefill)) {
        trimmedOutput = prefill + trimmedOutput;
        try {
            const parsedJSON = JSON.parse(trimmedOutput);
            console.warn('Assistant prefill was missing and has been added.');
            return parsedJSON;
        } catch (error) {
            console.warn('Unable to parse JSON with prefill.');
        }
    }

    try {
        // Case: JSON wrapped in code blocks
        const codeBlockRegex = /```(?:json)?\s*(\{[\s\S]*?\})\s*```/;
        const codeBlockMatch = output.match(codeBlockRegex);

        if (codeBlockMatch && codeBlockMatch[1]) {
            return JSON.parse(codeBlockMatch[1]);
        }

        // Case: Plain JSON without code blocks
        const jsonRegex = /(\{[\s\S]*?\})/;
        const jsonMatch = output.match(jsonRegex);

        if (jsonMatch && jsonMatch[1]) {
            return JSON.parse(jsonMatch[1]);
        }

        throw new Error('No JSON pattern found in the output');
    } catch (error) {
        console.error('Failed to extract JSON using regex methods:', error);

        // Last resort: brute force approach
        try {
            // Find the first opening brace and last closing brace
            const firstBrace = output.indexOf('{');
            const lastBrace = output.lastIndexOf('}');

            if (firstBrace !== -1 && lastBrace !== -1 && firstBrace < lastBrace) {
                const possibleJson = output.substring(firstBrace, lastBrace + 1);
                return JSON.parse(possibleJson);
            }
        } catch (error) {
            console.error('Brute force JSON extraction failed:', error);
        }

        throw new Error('Unable to extract valid JSON from the output');
    }
}

async function extractTextFromTag(str, tag) {
    const regex = new RegExp(`<${tag}>(.*?)</${tag}>`, 's');
    const match = str.match(regex);

    if (match) {
        return match[1].trim();
    } else {
        const openingTagIndex = str.indexOf(`<${tag}>`);
        if (openingTagIndex !== -1) {
            console.warn(`Warning: Closing tag </${tag}> not found. Returning all text after opening tag.

${str}`);
            return str.slice(openingTagIndex + tag.length + 2).trim();
        } else {
            console.error(`Error: Opening tag <${tag}> not found.

${str}`);
            return '###';
        }
    }
}

// Utility function for context retrieval
function getPreviousContext(map, currentId, n = 5) {
    const entries = Array.from(map.entries());
    const currentIndex = entries.findIndex(([id]) => id === currentId);

    // If ID not found or it's the first entry, return empty array
    if (currentIndex === -1 || currentIndex === 0) {
        return [];
    }

    const startIndex = Math.max(0, currentIndex - n);
    return entries
        .slice(startIndex, currentIndex)
        .map(([_, text]) => text);
}

// Utility function for content replacement
function updateParagraphContent(id, newContent) {
    const paragraph = document.querySelector(`p[id="${id}"]`);
    if (paragraph) {
        paragraph.textContent = newContent;
    }
}

// ================================================================================
// Main translation logic
// ================================================================================

async function translateWebpage() {
    try {
        const texts = extractTextFromWebpage();

        // Update only if models are selected
        if (models.stage1a.model !== '' && (models.stage1b.model !== '')) {
            const existingDict = loadDictionary();
            const newUpdates = await generateDictionary(texts);
            const updatedDict = await updateDictionary(existingDict, newUpdates);
            // Save to storage
            saveDictionary(updatedDict);
            // Update global variable
            dictionary = structuredClone(updatedDict);
        } else {
            // Use saved state
            dictionary = loadDictionary();
        }

        const entries = Array.from(texts.entries());

        const translationQueue = new TranslationQueueWithRetry();
        translationQueue.setTextDisplay('Translating text ...');
        translationQueue.setProgressDisplay(progressDisplay, entries.length);

        const translationPromises = entries.map(([currentId, currentText]) => {
            return translationQueue.enqueueTask(async () => {
                const previousContext = getPreviousContext(texts, currentId, 5);
                const translatedText = await translateLine(previousContext, currentText, dictionary);
                updateParagraphContent(currentId, translatedText);
                return { id: currentId, translation: translatedText };
            });
        });

        await Promise.allSettled(translationPromises);
    } catch (error) {
        console.error('Translation pipeline error:', error);
        errorHandler();
    }
}

// ================================================================================
// Text extraction logic
// ================================================================================

class TextPreProcessor {
    constructor(text) {
        this.text = text;
    }

    normalizeText() {
        this.text = this.text.normalize('NFKC');
        return this;
    }

    // Moves contents in ruby annotation to a inside brackets, placed at right side of original location
    processRubyAnnotations() {
        const temp = document.createElement('div');
        temp.innerHTML = this.text;

        temp.querySelectorAll('ruby').forEach(ruby => {
            const baseText = ruby.textContent.replace(/\s+/g, '');
            const rtText = ruby.querySelector('rt')?.textContent || '';
            ruby.textContent = `${baseText}(${rtText})`;
        });

        this.text = temp.textContent;
        return this;
    }

    removeBrTags() {
        this.text = this.text.replace(/<br\s*\/?>/gi, '');
        return this;
    }

    removeNonTextChars() {
        const pattern = new RegExp('[　・◇◆\\.]+', 'g');
        this.text = this.text.replace(pattern, '');
        return this;
    }

    trim() {
        this.text = this.text.trim();
        return this;
    }

    getText() {
        return this.text;
    }
}

function extractTextFromWebpage() {
    const config = domainManager.currentConfig;
    return config.extractText();
}

// ================================================================================
// API call parallelization
// ================================================================================

class TranslationQueue {
    constructor(concurrencyLimit = 10, requestDelay = 500) {
        this.activePromises = new Set();
        this.concurrencyLimit = concurrencyLimit;
        this.requestDelay = requestDelay;
        this.lastRequestTime = 0;
        this.taskQueue = [];
        this.progressMetrics = null;
        this.progressElement = null;
        this.processQueue = this.processQueue.bind(this);
        this.description = null;
    }

    setProgressDisplay(element, totalTasks) {
        this.progressElement = element;
        this.progressMetrics = new ProgressMetrics(totalTasks);
        this.updateProgressDisplay();
    }

    setTextDisplay(message) {
        this.description = message;
    }

    updateProgressDisplay() {
        if (!this.progressElement || !this.progressMetrics) return;

        const metrics = this.progressMetrics.getMetrics();
        this.progressElement.textContent = [
            this.description,
            `Progress: ${metrics.progress}`,
            `Time: ${metrics.timeMetrics}`,
            `Speed: ${metrics.speed}`
        ].filter(Boolean).join('\n'); // Exclude description if empty

        if (this.progressMetrics.completedTasks < this.progressMetrics.totalTasks) {
            requestAnimationFrame(() => this.updateProgressDisplay());
        }
    }

    async enqueueTask(task) {
        return new Promise((resolve, reject) => {
            this.taskQueue.push({ task, resolve, reject });
            this.processQueue();
        });
    }

    async processQueue() {
        if (this.activePromises.size >= this.concurrencyLimit || !this.taskQueue.length) {
            return;
        }

        const timeSinceLastRequest = Date.now() - this.lastRequestTime;
        if (timeSinceLastRequest < this.requestDelay) {
            setTimeout(() => this.processQueue(), this.requestDelay - timeSinceLastRequest);
            return;
        }

        const { task, resolve, reject } = this.taskQueue.shift();
        const promise = (async () => {
            try {
                this.lastRequestTime = Date.now();
                const result = await task();
                if (this.progressMetrics) {
                    this.progressMetrics.update();
                }
                resolve(result);
            } catch (error) {
                reject(error);
            } finally {
                this.activePromises.delete(promise);
                this.processQueue();
            }
        })();

        this.activePromises.add(promise);
    }

    async waitForCompletion() {
        while (this.activePromises.size > 0 || this.taskQueue.length > 0) {
            await Promise.race([...this.activePromises]);
        }
    }
}

// Adds retry upon failure, e.g. HTTP 429
class TranslationQueueWithRetry extends TranslationQueue {
    constructor(concurrencyLimit = 10, requestDelay = 500, maxRetries = 2) {
        super(concurrencyLimit, requestDelay);
        this.maxRetries = maxRetries;
    }

    async enqueueTask(task, retryCount = 0) {
        return new Promise((resolve, reject) => {
            const wrappedTask = async () => {
                try {
                    return await task();
                } catch (error) {
                    if (retryCount < this.maxRetries) {
                        return await this.enqueueTask(task, retryCount + 1);
                    }
                    throw error;
                }
            };
            this.taskQueue.push({ task: wrappedTask, resolve, reject });
            this.processQueue();
        });
    }
}

class ProgressMetrics {
    constructor(totalTasks) {
        this.startTime = Date.now();
        this.totalTasks = totalTasks;
        this.completedTasks = 0;
        this.lastUpdateTime = this.startTime;
        this.taskTimestamps = [];
        this.movingWindowSize = 20;
    }

    update() {
        this.completedTasks++;
        const currentTime = Date.now();
        this.taskTimestamps.push(currentTime);
        if (this.taskTimestamps.length > this.movingWindowSize) {
            this.taskTimestamps.shift();
        }
        this.lastUpdateTime = currentTime;
    }

    getMetrics() {
        const currentTime = Date.now();
        const elapsedSeconds = (currentTime - this.startTime) / 1000;

        // Calculate moving average speed
        const recentTasks = this.taskTimestamps.length;
        const windowTimespan = recentTasks > 1
        ? (this.taskTimestamps[recentTasks - 1] - this.taskTimestamps[0]) / 1000
        : elapsedSeconds;
        const currentSpeed = recentTasks > 1
        ? recentTasks / windowTimespan
        : this.completedTasks / elapsedSeconds;

        // Estimate remaining time
        const remainingTasks = this.totalTasks - this.completedTasks;
        const estimatedRemainingSeconds = currentSpeed > 0
        ? remainingTasks / currentSpeed
        : 0;

        return {
            progress: `${this.completedTasks}/${this.totalTasks}`,
            timeMetrics: `${this.formatTime(elapsedSeconds)}<${this.formatTime(estimatedRemainingSeconds)}`,
            speed: `${currentSpeed.toFixed(2)}t/s`
        };
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
}

// ================================================================================
// API keys management
// ================================================================================

function loadStoredState() {
    OPENROUTER_API_KEY = GM_getValue(STORAGE_KEYS.OPENROUTER, '');
    ANTHROPIC_API_KEY = GM_getValue(STORAGE_KEYS.ANTHROPIC, '');
    OPENAI_API_KEY = GM_getValue(STORAGE_KEYS.OPENAI, '');
    currentProvider = GM_getValue(STORAGE_KEYS.PROVIDER, 'OPENROUTER');
}

// ================================================================================
// Multi domain support
// ================================================================================

const DOMAIN_CONFIGS = {
    'syosetu.com': {
        domainCode: '1',
        getSeriesId: (url) => {
            const match = url.match(/^\/([^/]+)/);
            return match ? match[1] : null;
        },
        extractText: () => {
            const textContainers = document.querySelectorAll('.js-novel-text');
            if (!textContainers.length) return new Map();

            const paragraphMap = new Map();
            textContainers.forEach(container => {
                const pElements = Array.from(container.querySelectorAll('p'));

                pElements.forEach(p => {
                    const processedText = new TextPreProcessor(p.textContent)
                    .normalizeText()
                    .processRubyAnnotations()
                    .removeBrTags()
                    .removeNonTextChars()
                    .trim()
                    .getText();

                    if (processedText && p.id) {
                        paragraphMap.set(p.id, processedText);
                    }
                });
            });
            return paragraphMap;
        }
    },
    'kakuyomu.jp': {
        domainCode: '2',
        getSeriesId: (url) => {
            const match = url.match(/\/works\/(\d+)/);
            return match ? match[1] : null;
        },
        extractText: () => {
            const container = document.querySelectorAll('.episode-content');
            if (!container) return new Map();

            const paragraphMap = new Map();
            const pElements = Array.from(container.querySelectorAll('p'));

            pElements.forEach((p, index) => {
                const processedText = new TextPreProcessor(p.textContent)
                .normalizeText()
                .processRubyAnnotations()
                .removeBrTags()
                .removeNonTextChars()
                .trim()
                .getText();

                // xyz.jp doesn't use IDs, so we generate our own
                const id = `p-${index}`;
                if (processedText) {
                    paragraphMap.set(id, processedText);
                }
            });
            return paragraphMap;
        }
    }
};

class DomainManager {
    constructor() {
        this.currentConfig = this.detectDomain();
    }

    detectDomain() {
        const hostname = window.location.hostname;
        for (const [domain, config] of Object.entries(DOMAIN_CONFIGS)) {
            if (hostname.endsWith(domain)) {
                return {
                    ...config,
                    domain
                };
            }
        }
        throw new Error('Unsupported domain');
    }

    getCurrentSeriesId() {
        return this.currentConfig.getSeriesId(window.location.pathname);
    }

    getDictionaryKey(seriesId) {
        return `metadataDict_${seriesId}_${this.currentConfig.domainCode}`;
    }
}

// ================================================================================
// Main UI
// ================================================================================

function createModelSelector(options, modelKey) {
    let selector = document.createElement('select');
    selector.style.cssText = `
        margin: 0px;
        padding: 0px;
    `;

    selector.dataset.modelKey = modelKey;

    options.forEach(option => {
        let optionElement = document.createElement('option');
        optionElement.value = option.value ? JSON.stringify(option.value) : '';
        optionElement.textContent = option.label;
        selector.appendChild(optionElement);
    });

    selector.addEventListener('change', function() {
        if (this.value) {
            const selectedValue = JSON.parse(this.value);
            models[modelKey] = selectedValue;
        } else {
            models[modelKey] = {'model': '', 'providers': ''};
        }
    });

    return selector;
}

function createProviderSelector() {
    const div = document.createElement('div');
    const selector = document.createElement('select');
    div.style.cssText = `
        margin: 0px;
        padding: 0px;
    `;

    Object.values(PROVIDER_CONFIGS).forEach(provider => {
        const option = document.createElement('option');
        option.value = provider.id.toUpperCase();
        option.textContent = provider.label;
        option.selected = currentProvider === provider.id.toUpperCase();
        selector.appendChild(option);
    });

    selector.addEventListener('change', function() {
        currentProvider = this.value;
        GM_setValue(STORAGE_KEYS.PROVIDER, currentProvider);

        // Reset models object
        Object.keys(models).forEach(key => {
            models[key] = {'model': '', 'providers': ''};
        });

        // Update all model selectors
        updateModelSelectors();
    });

    div.append((() => {
        const p = document.createElement('p');
        p.textContent = 'Select Provider';
        Object.assign(p.style, {
            fontSize: '14px',
            padding: '2px',
            margin: '0px'
        });
        return p;
    })());
    div.appendChild(selector);

    return div;
}

function updateModelSelectors() {
    const providerConfig = PROVIDER_CONFIGS[currentProvider];

    // Update each stage selector with new options
    document.querySelectorAll('select[data-model-key]').forEach(selector => {
        const modelKey = selector.dataset.modelKey;
        // Map stage1a -> S1A_OPTIONS, stage1b -> S1B_OPTIONS, stage2 -> S2_OPTIONS
        const stageKey = modelKey
        .replace(/^stage(\d+)([a-b])?/, (_, num, letter) =>
                 `S${num}${letter ? letter.toUpperCase() : ''}_OPTIONS`
            );
        const options = providerConfig[stageKey];

        if (!options) {
            console.error(`Invalid stageKey: ${stageKey} for provider: ${currentProvider}`);
            return;
        }

        // Clear existing options
        selector.innerHTML = '';

        // Add new options
        options.forEach(option => {
            let optionElement = document.createElement('option');
            optionElement.value = option.value ? JSON.stringify(option.value) : '';
            optionElement.textContent = option.label;
            selector.appendChild(optionElement);
        });
    });
}

// ================================================================================
// Dialog box UI
// ================================================================================

function createKeyManagementDialog() {
    const dialog = document.createElement('div');
    dialog.style.cssText = `
        position: fixed;
        top: 15%;
        left: 15%;
        width: 70%;
        height: 70%;
        background-color: white;
        border: 1px solid #ccc;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 10000;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 15px;
    `;

    const createInput = (label, value) => {
        const container = document.createElement('div');
        container.style.cssText = 'display: flex; flex-direction: column; gap: 5px; align-items: center;';

        const labelElement = document.createElement('label');
        labelElement.textContent = label;

        const input = document.createElement('input');
        input.type = 'password';
        input.value = value || '';
        input.style.cssText = 'padding: 8px; width: 100%; font-family: monospace;';

        container.appendChild(labelElement);
        container.appendChild(input);
        return { container, input };
    };

    const inputs = {
        openrouter: createInput('OpenRouter API Key:', OPENROUTER_API_KEY),
        anthropic: createInput('Anthropic API Key:', ANTHROPIC_API_KEY),
        openai: createInput('OpenAI API Key:', OPENAI_API_KEY)
    };

    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = `
        display: flex;
        gap: 20px;
        margin-top: auto;
        justify-content: center;
        padding-bottom: 30px;
    `;

    const okButton = document.createElement('button');
    okButton.textContent = 'OK';
    okButton.style.cssText = `
        padding: 10px 20px;
        font-size: 1.5rem;
        min-width: 250px;
        cursor: pointer;
        border-radius: 4px;
        border: 1px solid #ccc;
    `;
    okButton.onclick = () => {
        // Save non-empty values
        const newKeys = {
            [STORAGE_KEYS.OPENROUTER]: inputs.openrouter.input.value,
            [STORAGE_KEYS.ANTHROPIC]: inputs.anthropic.input.value,
            [STORAGE_KEYS.OPENAI]: inputs.openai.input.value
        };

        Object.entries(newKeys).forEach(([key, value]) => {
            if (value.trim()) {
                GM_setValue(key, value);
            }
        });

        // Update global variables
        loadStoredState();
        dialog.remove();
    };

    const cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.style.cssText = `
        padding: 10px 20px;
        font-size: 1.5rem;
        min-width: 250px;
        cursor: pointer;
        border-radius: 4px;
        border: 1px solid #ccc;
    `;
    cancelButton.onclick = () => dialog.remove();

    buttonContainer.appendChild(okButton);
    buttonContainer.appendChild(cancelButton);

    // Append all elements to dialog
    Object.values(inputs).forEach(({ container }) => dialog.appendChild(container));
    dialog.appendChild(buttonContainer);

    return dialog;
}

function createDatabaseEditorDialog() {
    const dialog = document.createElement('div');
    dialog.style.cssText = `
        position: fixed;
        top: 15%;
        left: 15%;
        width: 70%;
        height: 70%;
        background-color: white;
        border: 1px solid #ccc;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        padding: 20px;
        display: flex;
        flex-direction: column;
        gap: 15px;
    `;

    // Create scrollable container for entries
    const entriesContainer = document.createElement('div');
    entriesContainer.style.cssText = `
        flex: 1;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        gap: 10px;
    `;

    // Load dictionary and create entry boxes
    const dictionary = loadDictionary();
    dictionary.entries.forEach(entry => {
        const entryBox = createEntryBox(entry);
        entriesContainer.appendChild(entryBox);
    });

    // Create button container
    const buttonContainer = document.createElement('div');
    buttonContainer.style.cssText = `
        display: flex;
        justify-content: center;
        gap: 20px;
        padding: 10px;
    `;

    const saveButton = document.createElement('button');
    saveButton.textContent = 'Save Changes';
    saveButton.style.cssText = `
        padding: 10px 20px;
        font-size: 1.5rem;
        min-width: 250px;
        cursor: pointer;
        border-radius: 4px;
        border: 1px solid #ccc;
        background-color: #0077cc;
        color: white;
    `;
    saveButton.onclick = () => {
        // Get all entry boxes
        const entryBoxes = Array.from(entriesContainer.children);

        // Collect all keys to check for duplicates
        const allKeys = new Set();
        let hasError = false;
        let errorMessage = '';

        // Validate and collect entries
        const newEntries = entryBoxes.map((box, index) => {
            const state = box.getState();

            // Check for empty keys
            if (state.keys.length === 0) {
                hasError = true;
                errorMessage = 'Each entry must have at least one key. Please add keys to continue. See entry with value:' + state.value;
                return null;
            }

            // Check for empty value
            if (!state.value.trim()) {
                hasError = true;
                errorMessage = `Entry value cannot be empty. Please add content to continue. See entry with keys: ${state.value}`;
                return null;
            }

            // Remove duplicate keys silently
            const uniqueKeys = state.keys.filter(key => {
                if (allKeys.has(key)) {
                    return false;
                }
                allKeys.add(key);
                return true;
            });

            return {
                id: index + 1, // Generate sequential IDs
                keys: uniqueKeys,
                value: state.value
            };
        });

        if (hasError) {
            // Create and show error dialog
            const errorDialog = document.createElement('div');
            errorDialog.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                z-index: 1001;
            `;

            errorDialog.innerHTML = `
                <p style="margin-bottom: 15px;">${errorMessage}</p>
                <button style="padding: 5px 15px;">OK</button>
            `;
            document.body.appendChild(errorDialog);

            const okButton = errorDialog.querySelector('button');
            okButton.onclick = () => {
                errorDialog.remove();
            };

            return;
        }

        // Save the updated dictionary
        saveDictionary({
            entries: newEntries
        });

        dialog.remove();
    };

    const cancelButton = document.createElement('button');
    cancelButton.textContent = 'Cancel';
    cancelButton.style.cssText = `
        padding: 10px 20px;
        font-size: 1.5rem;
        min-width: 250px;
        cursor: pointer;
        border-radius: 4px;
        border: 1px solid #ccc;
        background-color: #666;
        color: white;
    `;
    cancelButton.onclick = () => {
        saveDictionary(dictionary);
        dialog.remove();
    };

    buttonContainer.appendChild(saveButton);
    buttonContainer.appendChild(cancelButton);

    // Assemble the dialog
    dialog.appendChild(entriesContainer);
    dialog.appendChild(buttonContainer);

    return dialog;
}

function createEntryBox(entry) {
    const entryBox = document.createElement('div');
    entryBox.style.cssText = `
        border: 1px solid #ccc;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #fff;
        width: 100%;
        display: flex;
        flex-direction: column;
        gap: 15px;
    `;

    // Create keys container with flex-wrap
    const keysContainer = document.createElement('div');
    keysContainer.style.cssText = `
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: start;
        width: 100%;
    `;

    // Store the current keys
    const currentKeys = [...entry.keys];

    function createKeyBox(keyValue = '', index = null) {
        const keyWrapper = document.createElement('div');
        keyWrapper.style.cssText = `
            display: flex;
            align-items: center;
            border: 1px solid #0077cc;
            padding: 2px;
            gap: 5px;
            background-color: #fff;
        `;

        const keyInput = document.createElement('input');
        keyInput.type = 'text';
        keyInput.value = keyValue;
        keyInput.style.cssText = `
            padding: 5px;
            min-width: 100px;
            border: none;
            outline: none;
        `;

        // Track changes to update currentKeys
        keyInput.addEventListener('input', () => {
            if (index !== null) {
                currentKeys[index] = keyInput.value;
            }
        });

        const deleteBtn = document.createElement('button');
        deleteBtn.innerHTML = '✕';
        deleteBtn.style.cssText = `
            background: #0077cc;
            color: white;
            border: none;
            width: 20px;
            height: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            font-size: 12px;
            padding: 0;
        `;

        deleteBtn.onclick = () => {
            keyWrapper.remove();
            if (index !== null) {
                currentKeys.splice(index, 1);
            }
        };

        keyWrapper.appendChild(keyInput);
        keyWrapper.appendChild(deleteBtn);
        return keyWrapper;
    }

    // Add existing keys
    entry.keys.forEach((key, index) => {
        keysContainer.appendChild(createKeyBox(key, index));
    });

    // Add button
    const addButton = document.createElement('button');
    addButton.innerHTML = '+';
    addButton.style.cssText = `
        width: 30px;
        height: 30px;
        background: #0077cc;
        color: white;
        border: none;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        font-size: 20px;
    `;

    addButton.onclick = () => {
        const newKeyBox = createKeyBox('', currentKeys.length);
        keysContainer.insertBefore(newKeyBox, addButton);
        currentKeys.push('');
        const input = newKeyBox.querySelector('input');
        input.focus();
    };

    keysContainer.appendChild(addButton);

    // Create value textarea
    const valueContainer = document.createElement('div');
    valueContainer.style.width = '100%';

    const valueTextarea = document.createElement('textarea');
    valueTextarea.value = entry.value;
    valueTextarea.style.cssText = `
        width: 100%;
        min-height: 100px;
        padding: 10px;
        border: 1px solid #ccc;
        resize: vertical;
    `;

    valueContainer.appendChild(valueTextarea);

    // Add everything to the entry box
    entryBox.appendChild(keysContainer);
    entryBox.appendChild(valueContainer);

    // Add method to get current state
    entryBox.getState = () => ({
        id: entry.id,
        keys: currentKeys.filter(k => k.trim() !== ''),
        value: valueTextarea.value
    });

    return entryBox;
}

// ================================================================================
// Init script
// ================================================================================

function init() {
    domainManager = new DomainManager();

    // Load stored keys on initialization
    loadStoredState();

    // Button to collapse floating box UI
    const collapseButton = document.createElement('button');
    collapseButton.textContent = '▼'; // Down arrow when expanded
    collapseButton.style.cssText = `
        float: left;
        position: absolute;
        top: 5px;
        right: 5px;
        border: none;
        background: none;
        cursor: pointer;
        padding: 5px;
        font-size: 12px;
    `;

    // Create the mini button (initially hidden)
    const miniButton = document.createElement('div');
    miniButton.style.cssText = `
        position: fixed;
        top: 60px;
        left: 3px;
        width: 30px;
        height: 30px;
        background-color: white;
        border: 1px solid #ccc;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        display: none;
        justify-content: center;
        align-items: center;
        cursor: pointer;
        z-index: 1000;
    `;
    miniButton.textContent = '▲'; // Up arrow when collapsed
    document.body.appendChild(miniButton);

    // Create the floating box
    const floatingBox = document.createElement('div');
    floatingBox.style.cssText = `
        position: fixed;
        top: 60px;
        left: 3px;
        z-index: 1000;
        padding: 25px 10px 10px 10px;  // Extra padding on top for collapse button
        background-color: white;
        border: 1px solid #ccc;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
        gap: 5px;
    `;

    // Add the collapse button to floating box
    floatingBox.appendChild(collapseButton);
    document.body.appendChild(floatingBox);

    // Toggle function
    function toggleUI() {
        if (floatingBox.style.display !== 'none') {
            floatingBox.style.display = 'none';
            miniButton.style.display = 'flex';
        } else {
            floatingBox.style.display = 'flex';
            miniButton.style.display = 'none';
        }
    }
    collapseButton.addEventListener('click', toggleUI);
    miniButton.addEventListener('click', toggleUI);

    // Add API key management button
    const manageKeysButton = document.createElement('button');
    manageKeysButton.textContent = 'Manage API Keys';
    manageKeysButton.onclick = () => {
        const dialog = createKeyManagementDialog();
        document.body.appendChild(dialog);
    };
    floatingBox.appendChild(manageKeysButton);

    // Add database editor button
    const dbEditorButton = document.createElement('button');
    dbEditorButton.textContent = 'Edit Metadata';
    dbEditorButton.onclick = () => {
        const dialog = createDatabaseEditorDialog();
        document.body.appendChild(dialog);
    };
    floatingBox.appendChild(dbEditorButton);

    // Add provider selector
    const providerSelector = createProviderSelector();
    floatingBox.appendChild(providerSelector);

    // Create stage selectors with initial provider's options
    const providerConfig = PROVIDER_CONFIGS[currentProvider];

    // Add model selectors to floating box
    const modelSelector = document.createElement('div');

    const stage1aSelector = document.createElement('div');
    stage1aSelector.append((() => {
        const p = document.createElement('p');
        p.textContent = ' Glossary Generate';
        Object.assign(p.style, {
            fontSize: '14px',
            padding: '2px',
            margin: '0px'
        });
        return p;
    })());
    stage1aSelector.appendChild(createModelSelector(providerConfig.S1A_OPTIONS, 'stage1a'));

    const stage1bSelector = document.createElement('div');
    stage1bSelector.append((() => {
        const p = document.createElement('p');
        p.textContent = 'Glossary Update';
        Object.assign(p.style, {
            fontSize: '14px',
            padding: '2px',
            margin: '0px'
        });
        return p;
    })());
    stage1bSelector.appendChild(createModelSelector(providerConfig.S1B_OPTIONS, 'stage1b'));

    const stage2Selector = document.createElement('div');
    stage2Selector.append((() => {
        const p = document.createElement('p');
        p.textContent = 'Translation';
        Object.assign(p.style, {
            fontSize: '14px',
            padding: '2px',
            margin: '0px'
        });
        return p;
    })());
    stage2Selector.appendChild(createModelSelector(providerConfig.S2_OPTIONS, 'stage2'));

    modelSelector.appendChild(stage1aSelector);
    modelSelector.appendChild(stage1bSelector);
    modelSelector.appendChild(stage2Selector);
    floatingBox.appendChild(modelSelector);

    // Add button to floating box
    const button = document.createElement('button');
    button.innerText = 'Translate';
    button.style.marginRight = '10px';
    floatingBox.appendChild(button);
    button.addEventListener('click', function() {
        if (models.stage2.model !== '') {
            translateWebpage();
        } else {
            errorDisplay.textContent = 'Please select a model before translating.';
            errorDisplay.style.display = 'block';
        }
    });

    // Progress display
    progressDisplay = document.createElement('div');
    progressDisplay.style.fontFamily = 'monospace';
    progressDisplay.style.whiteSpace = 'pre';
    progressDisplay.style.marginTop = '5px';
    floatingBox.appendChild(progressDisplay);

    // Error display
    errorDisplay = document.createElement('div');
    errorDisplay.style.color = 'red';
    errorDisplay.style.marginTop = '5px';
    errorDisplay.style.display = 'none';
    floatingBox.appendChild(errorDisplay);
}

window.addEventListener('load', init);
