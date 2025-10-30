// ==UserScript==
// @name         Webpage Text Translator
// @author       Qing Wen
// @homepage     https://github.com/qw02/llm-translate-userscript
// @namespace    https://github.com/qw02
// @version      2.2
// @description  Translates webnovels with LLM using a RAG pipeline.
// @match        https://*.syosetu.com/*/*/
// @match        https://kakuyomu.jp/works/*/episodes/*
// @match        file:///*
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM.xmlHttpRequest
// ==/UserScript==

'use strict';

/* ================================================================================
   Constants
   ================================================================================*/

// External API Rate limits
const MAX_CONCURRENCY = 10;
const MAX_CALLS_PER_SEC = 10;
const MAX_RETRIES = 3; // Per call retry. Counts all, including HTTP 429, server errors, etc.
const BASE_RETRY_DELAY_MS = 1000; // Simple exponential backoff

// Chunk size (num chars) for glossary generation. Low values can result in higher number of calls during merge process
// 1~5k works for large models (DS V3, GPT-5, Gemini2.5 Pro etc.)
// Reduce to 100~500 for small (<20b) models
const GLOSSARY_CHUNK_SIZE = 4000;

// Chunk size for segmentation for translation
const BATCH_CHAR_LIMIT = 1500;
// How many paragraphs to include from the end of the previous batch
const OVERLAP_PARAGRAPH_COUNT = 5;

// Used as prefix to HTML id names to prevent collision with existing on the page
const SCRIPT_PREFIX = 'us-123456-';

// Populated with avaliable models (those with API keys set) on load
const modelsList = [];

// Set to true to show test models in selectors
// These sent the call to an internal echo, see 'Test' section at bottom for details
const TEST_MODE = false;

// Info to log to console
const LOG_LLM = {
  system: false,
  user: true,
  reasoning: false,
  assistant: true,
}

/* ================================================================================
   Config
   ================================================================================*/

// API keys loaded during init from storage
const PROVIDER_API_CONFIG = {
  openrouter: {
    apiKey: null,
    endpoint: 'https://openrouter.ai/api/v1/chat/completions',
  },
  openai: {
    apiKey: null,
    endpoint: 'https://api.openai.com/v1/chat/completions',
  },
  anthropic: {
    apiKey: null,
    endpoint: 'https://api.anthropic.com/v1/messages',
  },
  deepseek: {
    apiKey: null,
    endpoint: 'https://api.deepseek.com/v1/chat/completions',
  },
  xai: {
    apiKey: null,
    endpoint: 'https://api.x.ai/v1/chat/completions',
  },
  google: {
    apiKey: null,
    endpoint: 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
  },
  nanogpt: {
    apiKey: null,
    endpoint: 'https://nano-gpt.com/api/v1/chat/completions',
  },
};

const PROVIDER_CONFIGS = {
  openrouter: {
    models: [
      { id: '1-1', model: 'moonshotai/kimi-k2-0905', label: 'Kimi K2 0905', 'providers': ['DeepInfra', 'Chutes'] },
      { id: '1-2', model: 'google/gemini-2.5-pro-preview', label: 'Gemini Pro 2.5', 'providers': ['Google AI Studio', 'Google'] },
      { id: '1-3', model: 'google/gemini-2.5-flash', label: 'Gemini Flash 2.5', 'providers': ['Google AI Studio', 'Google'] },
      { id: '1-4', model: 'google/gemini-2.5-flash-lite-preview-06-17', label: 'Gemini Flash-Lite 2.5', 'providers': ['Google AI Studio', 'Google'] },
      { id: '1-5', model: 'z-ai/glm-4-32b', label: 'GLM 4 32b', 'providers': ['z-ai'], tokens: 8192 },
      { id: '1-6', model: 'deepseek/deepseek-v3.2-exp', label: 'DeepSeek V3.2 (R)', 'providers': ['DeepInfra', 'DeepSeek', 'Novita'], reasoning: true, tokens: 8192 },
      { id: '1-7', model: 'x-ai/grok-4-fast', label: 'Grok 4 Fast (R)', 'providers': ['xAI'], reasoning: true, tokens: 8192 },
      { id: '1-8', model: 'x-ai/grok-4-fast', label: 'Grok 4 Fast', 'providers': ['xAI'], reasoning: false },
      { id: '1-9', model: 'z-ai/glm-4.6', label: 'GLM 4.6', 'providers': ['z-ai'], tokens: 8192 },
      { id: '1-10', model: 'anthropic/claude-sonnet-4.5', label: 'Sonnet 4.5' },
    ],
    limits: {
      stage1: 'all',
      stage2: ['1-4', '1-5', '1-8'],
      stage3a: ['1-1', '1-3', '1-6', '1-7', '1-8'],
      stage3b: 'all',
    },
  },
  openai: {
    models: [
      { id: '3-1', model: 'gpt-5', label: 'GPT-5 (R: Low)', reasoning: 'low' },
      { id: '3-2', model: 'gpt-5', label: 'GPT-5 (R: High)', reasoning: 'high', tokens: 8192 },
      { id: '3-3', model: 'gpt-5-mini', label: 'GPT-5 Mini (R: Off)', reasoning: 'minimal' },
      { id: '3-4', model: 'gpt-5-nano', label: 'GPT-5 Nano (R: Off)', reasoning: 'minimal' },
    ],
    limits: {
      stage1: 'all',
      stage2: ['3-4'],
      stage3a: ['3-3', '3-4'],
      stage3b: 'all',
    },
  },
  deepseek: {
    models: [
      { id: '4-1', model: 'deepseek-chat', label: 'DeepSeek V3.2 Exp (R: Off)', reasoning: false },
      { id: '4-3', model: 'deepseek-reasoner', label: 'DeepSeek V3.2 Exp (R: On)', reasoning: true, tokens: 8192 },
    ],
    limits: {
      stage1: 'all',
      stage2: ['4-1'],
      stage3a: ['4-1'],
      stage3b: 'all',
    },
  },
  xai: {
    models: [
      { id: '5-1', model: 'grok-4-fast-reasoning', label: 'Grok 4 Fast (R)', tokens: 8192 },
      { id: '5-2', model: 'grok-4-fast-non-reasoning', label: 'Grok 4 Fast' },
      { id: '5-3', model: 'grok-4', label: 'Grok 4', tokens: 8192 },
    ],
    limits: {
      stage1: 'all',
      stage2: ['5-2'],
      stage3a: ['5-1', '5-2'],
      stage3b: ['5-1', '5-2'],
    },
  },
  google: {
    models: [
      { id: '6-1', model: 'gemini-2.5-pro', label: 'Gemini Pro 2.5 (R: Med)', reasoning: 'medium', tokens: 8192 },
      { id: '6-2', model: 'gemini-2.5-pro', label: 'Gemini Pro 2.5 (R: Low)', reasoning: 'low' },
      { id: '6-3', model: 'gemini-2.5-flash-lite-preview-09-2025', label: 'Gemini Flash-Lite 2.5 (R: Med)', reasoning: 'medium', tokens: 8192 },
      { id: '6-4', model: 'gemini-2.5-flash-lite-preview-09-2025', label: 'Gemini Flash-Lite 2.5 (R: Off)', reasoning: 'minimal' },
      { id: '6-5', model: 'gemini-2.5-flash-preview-09-2025', label: 'Gemini Flash 2.5 (R: Med)', reasoning: 'medium', tokens: 8192 },
      { id: '6-6', model: 'gemini-2.5-flash-preview-09-2025', label: 'Gemini Flash 2.5 (R: Off)', reasoning: 'minimal' },
    ],
    limits: {
      stage1: 'all',
      stage2: ['6-4', '6-6'],
      stage3a: ['6-3', '6-4', '6-5', '6-6'],
      stage3b: 'all',
    },
  },
  anthropic: {
    models: [
      { id: '2-1', model: 'claude-sonnet-4-5', label: 'Sonnet 4.5' },
      { id: '2-2', model: 'claude-haiku-4-5', label: 'Haiku 4.5' },
    ],
    limits: {
      stage1: 'all',
      stage2: ['2-2'],
      stage3a: ['2-2'],
      stage3b: 'all',
    },
  },
  nanogpt: {
    models: [
      { id: '7-1', model: 'deepseek-ai/deepseek-v3.2-exp', label: '[NG] DeepSeek V3.2 (R: Off)' },
      { id: '7-2', model: 'deepseek-ai/deepseek-v3.2-exp-thinking', label: '[NG] DeepSeek V3.2 (R: On)', tokens: 8192  },
      { id: '7-3', model: 'moonshotai/Kimi-K2-Instruct-0905', label: '[NG] Kimi K2 0905' },
      { id: '7-4', model: 'z-ai/glm-4.6', label: '[NG] GLM 4.6' },
    ],
    limits: {
      stage1: 'all',
      stage2: ['7-1', '7-3'],
      stage3a: ['7-1', '7-3', '7-4'],
      stage3b: 'all',
    },
  },
};

/* ================================================================================
   LLM API
   ================================================================================*/

/**
 * Represents a specific LLM model from a provider, handling API interactions.
 * An instance of this class is created for each stage of the process,
 * encapsulating the provider, model, and API key details.
 */
class LLMClient {
  /**
   * Instantiates a new LLMClient for a specific model.
   * @param {string} providerKey - The key for the provider (e.g., 'openai', 'anthropic').
   * @param {object} modelConfig - The configuration object for the specific model from PROVIDER_CONFIGS.
   * @param {string} apiKey - The API key for the provider.
   * @param {string} apiEndpoint - The base URL for the provider's API.
   * @param {object} [adapterRegistry] - A registry of API adapters. This allows for easy testing and extension.
   */
  constructor(providerKey, modelConfig, apiKey, apiEndpoint, adapterRegistry = ApiAdapters) {
    this._provider = providerKey;
    this._modelId = modelConfig.model;
    this._modelConfig = modelConfig;
    this._apiKey = apiKey;
    this._endpoint = apiEndpoint;

    // Select the correct adapter for the provider.
    this._adapter = adapterRegistry[providerKey];

    if (!this._adapter) {
      throw new Error(`No adapter found for provider: ${providerKey}. A new one must be created.`);
    }
  }

  /**
   * Sends a completion request to the configured LLM. This is the main public method.
   * It orchestrates the entire process: building the request, sending it, and parsing the response.
   * @param {string} systemMessage - The system instruction for the completion.
   * @param {string} userMessage - The user's message for the completion.
   * @returns {Promise<string>} A promise that resolves to the LLM's completion text.
   */
  async completion(systemMessage, userMessage) {
    try {
      const messages = this._buildMessages(systemMessage, userMessage);

      const requestDetails = this._adapter.buildRequest(
        this._endpoint,
        this._modelId,
        messages,
        this._modelConfig,
        this._apiKey,
      );

      const response = await this._makeHttpRequest(requestDetails);
      const { completion, reasoning } = this._adapter.parseResponse(response);
      this._logInteraction(messages, reasoning, completion);

      return completion;

    } catch (error) {
      console.error(`LLMClient Error for ${this._provider}/${this._modelId}:`, error);
      throw error;
    }
  }

  /**
   * Constructs the messages array in the standard format.
   * @private
   */
  _buildMessages(systemMessage, userMessage) {
    const messages = [];
    if (systemMessage) {
      messages.push({ role: 'system', content: systemMessage });
    }
    messages.push({ role: 'user', content: userMessage });
    return messages;
  }

  /**
   * A generic wrapper for GM.xmlHttpRequest to handle network-level issues.
   * @private
   */
  async _makeHttpRequest({ url, headers, payload }) {
    // TEST MODE: Intercept test provider calls and return mock responses
    if (this._provider === 'test') {
      return new Promise((resolve) => {
        // Simulate network delay
        setTimeout(() => {
          const mockResponse = generateTestResponse(this._modelId, payload.messages);
          resolve(mockResponse);
        }, Math.random() * 50 + 200); // Simulated latency
      });
    }

    // PRODUCTION MODE: Normal HTTP request
    return new Promise((resolve, reject) => {
      GM.xmlHttpRequest({
        method: 'POST',
        url: url,
        headers: headers,
        data: JSON.stringify(payload),
        responseType: 'json',
        onload: (res) => {
          if (res.status >= 200 && res.status < 300) {
            resolve(res.response);
          } else {
            // Log HTTP error details for debugging
            console.error('HTTP Request Failed:', {
              url: url,
              status: res.status,
              statusText: res.statusText,
              response: res.response,
              responseText: res.responseText,
            });
            const error = new Error(`HTTP error! status: ${res.status} ${res.statusText}`);
            error.response = res;
            reject(error);
          }
        },
        onerror: () => {
          console.error('Network Request Failed:', { url: url });
          reject(new Error('Network error: request failed (no response received)'));
        },
        ontimeout: () => {
          console.error('Request Timed Out:', { url: url });
          reject(new Error('Request timed out'));
        },
      });
    });
  }

  /**
   * Logs for debugging.
   * @private
   */
  _logInteraction(messages, reasoning, completion) {
    const systemMsg = messages.find(m => m.role === 'system')?.content || '';
    const userMsg = messages.find(m => m.role === 'user')?.content || '';

    const sections = [];

    if (systemMsg && LOG_LLM.system) sections.push({ title: '[System]', content: systemMsg });

    if (userMsg && LOG_LLM.user) sections.push({ title: '[User]', content: userMsg });

    if (reasoning && LOG_LLM.reasoning) sections.push({ title: '[Reasoning]', content: reasoning });

    if (completion && LOG_LLM.assistant) sections.push({ title: '[Assistant]', content: completion });

    if (!sections.length) return;

    const logText = sections
      .map(section => `${section.title}:\n${section.content}`)
      .join('\n' + '-'.repeat(80) + '\n');

    console.log('='.repeat(80) + '\n' + logText);
  }
}

/**
 * Factory function to create an OpenAI-style API adapter.
 * This serves as the base for most providers.
 *
 * @param {object} overrides - Optional overrides for headers, payload, or response parsing.
 * @returns {object} An adapter object.
 */
function createOpenAIApiAdapter(overrides = {}) {
  const {
    modifyHeaders = (headers) => headers,
    modifyPayload = (payload, modelConfig) => payload,
    modifyResponse = (response) => ({
      completion: response.choices?.[0]?.message?.content ?? '',
      reasoning: response.choices?.[0]?.message?.reasoning ?? null,
    }),
  } = overrides;

  return {
    buildRequest(endpoint, modelId, messages, modelConfig, apiKey) {
      // Standard OpenAI payload
      let payload = {
        model: modelId,
        messages: messages,
        max_tokens: modelConfig.tokens ?? 4096,
        temperature: 0.6,
        top_p: 0.95,
      };

      // Provider-specific payload tweaks
      payload = modifyPayload(payload, modelConfig);

      // Construct request headers
      let headers = { 'Content-Type': 'application/json' };
      headers = modifyHeaders(headers);
      headers['Authorization'] = `Bearer ${apiKey}`;

      return {
        url: endpoint,
        headers: headers,
        payload: payload,
      };
    },

    parseResponse(response) {
      if (response.error) {
        throw new Error(`API Error: ${response.error.message || JSON.stringify(response.error)}`);
      }
      if (!response.choices || response.choices.length === 0) {
        throw new Error('Invalid response: missing choices.');
      }
      return modifyResponse(response);
    },
  };
}

const ApiAdapters = {
  openai: createOpenAIApiAdapter(),

  deepseek: createOpenAIApiAdapter(),

  openrouter: createOpenAIApiAdapter({
    modifyHeaders(headers, apiKey) {
      return {
        ...headers,
        'HTTP-Referer': 'https://github.com/qw02/llm-translate-userscript',
        'X-Title': 'Translation Userscript',
      };
    },

    modifyPayload(payload, modelConfig) {
      if (modelConfig.providers) {
        payload.provider = {
          order: modelConfig.providers,
          allow_fallbacks: false,
        }
      }

      // Handle reasoning config
      const reasoningConfig = modelConfig.reasoning;

      if (reasoningConfig !== undefined) { // config can have value: false
        const reasoningPayload = {};
        const configType = typeof reasoningConfig;

        if (configType === 'boolean') {
          reasoningPayload.enabled = true
        } else if (configType === 'number') {
          reasoningPayload.max_tokens = reasoningConfig;
        } else if (configType === 'string') {
          reasoningPayload.effort = reasoningConfig;
        } else {
          console.error(`Invalid reasoning config in OpenRouter: ${configType}`);
        }
        payload.reasoning = reasoningPayload;

        if (configType === 'boolean' && reasoningConfig === false) {
          delete payload.reasoning;
        }
      }

      return payload;
    },
  }),

  xai: createOpenAIApiAdapter({
    modifyPayload(payload, modelConfig) {
      payload.max_completion_tokens = payload.max_tokens;
      delete payload.max_tokens;
      if (modelConfig.reasoning) {
        payload.reasoning_mode = modelConfig.reasoning;
      }

      return payload;
    },
  }),

  google: createOpenAIApiAdapter({
    modifyPayload(payload, modelConfig) {
      const reasoningBudgetMap = {
        'minimal': 0,
        'low': 128,
        'medium': 2048,
        'high': 8192,
      };

      if (reasoningBudgetMap[modelConfig.reasoning] !== undefined) {
        payload.extra_body = {
          google: {
            thinking_config: {
              thinking_budget: reasoningBudgetMap[modelConfig.reasoning],
              include_thoughts: true,
            },
          },
        };
      }

      return payload;
    },
  }),

  anthropic: {
    buildRequest(endpoint, modelId, messages, modelConfig, apiKey) {
      const systemMessage = messages.find(m => m.role === 'system')?.content || '';
      const userMessages = messages.filter(m => m.role !== 'system');

      return {
        url: endpoint,
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': apiKey,
          'anthropic-version': '2023-06-01',
          ...(systemMessage ? { 'anthropic-beta': 'prompt-caching-2024-07-31' } : {}),
        },
        payload: {
          model: modelId,
          messages: userMessages,
          system: systemMessage,
          max_tokens: 8192,
          temperature: 0.5,
        },
      };
    },
    parseResponse(response) {
      if (response.type === 'error') {
        throw new Error(`Anthropic API Error: ${response.error?.message}`);
      }
      if (!response.content?.[0]?.text) {
        throw new Error('Invalid Anthropic response.');
      }
      return { completion: response.content[0].text, reasoning: null };
    },
  },

  nanogpt: createOpenAIApiAdapter(),

  test: createOpenAIApiAdapter(),
};

/**
 * Helper function to find the full model configuration and its provider key
 * given a model ID from the UI selection.
 * @param {string} modelId - The ID selected by the user.
 * @returns {{ providerKey: string, modelConfig: object }} The provider key and model config.
 * @throws {Error} If the model ID is not found.
 */
function findModelConfigById(modelId) {
  for (const [providerKey, providerData] of Object.entries(PROVIDER_CONFIGS)) {
    const modelConfig = providerData.models.find(model => model.id === modelId);
    if (modelConfig) {
      return { providerKey, modelConfig };
    }
  }
  throw new Error(`No model found with ID: ${modelId}. Please check your selection.`);
}

/**
 * Factory function to create an LLMClient instance for a given stage.
 * @param {string} modelId - The unique model ID from the config.
 * @returns {LLMClient} A configured LLMClient instance.
 */
function createLLMClientForStage(modelId) {
  if (modelId == null) {
    throw new Error(`No model selected for createLLMClientForStage().`);
  }

  const { providerKey, modelConfig } = findModelConfigById(modelId);
  const apiConfig = PROVIDER_API_CONFIG[providerKey];

  if (!apiConfig) {
    throw new Error(`API configuration missing for provider: ${providerKey}. Please set your API key.`);
  }

  if (!ApiAdapters[providerKey]) {
    throw new Error(`No API adapter registered for provider: ${providerKey}.`);
  }

  return new LLMClient(
    providerKey,
    modelConfig,
    apiConfig.apiKey,
    apiConfig.endpoint,
    // The adapter registry injected by LLMClient constructor
  );
}

// Helper to create a stage queue and its metrics
function createStageQueue(modelId, label) {
  const client = createLLMClientForStage(modelId);
  const progressSection = document.getElementById('progress-section')
  const metrics = new ProgressMetrics({ parent: progressSection, label });
  metrics.setInitialTasks(0);
  return new RequestQueue(client, metrics);
}

/* ================================================================================
   Request queue and Progress monitoring
   ================================================================================*/

/**
 * A self-contained progress tracker that creates its own UI:
 * - Details panel (live updates while running)
 * - Summary bar (shown after completion, collapsible via + / -)
 */
class ProgressMetrics {
  constructor({ parent, label }) {
    this.label = label;
    this.total = 0;
    this.completed = 0;
    this.errors = 0;
    this.startTime = null;
    this.endTime = null;
    this._uiTimer = null;
    this._collapsed = false;

    // Build UI
    this.root = document.createElement('div');
    this.root.style.margin = '6px 0';
    this.root.style.border = '1px solid #ddd';
    this.root.style.borderRadius = '6px';
    this.root.style.background = '#fafafa';

    // Summary (hidden until done)
    this.summaryBar = document.createElement('div');
    this.summaryBar.style.display = 'none';
    this.summaryBar.style.padding = '6px 8px';
    this.summaryBar.style.display = 'none';
    this.summaryBar.style.fontFamily = 'monospace';
    this.summaryBar.style.display = 'none';
    this.summaryBar.style.alignItems = 'center';
    this.summaryBar.style.justifyContent = 'space-between';
    this.summaryBar.style.gap = '8px';
    this.summaryBar.style.display = 'none';
    this.summaryBar.style.borderBottom = '1px solid #eee';
    this.summaryBar.style.display = 'none';

    // Use flex properly only once
    this.summaryBar.style.display = 'none';
    this.summaryBar.style.display = 'flex';

    this.toggleBtn = document.createElement('button');
    this.toggleBtn.textContent = '＋';
    this.toggleBtn.style.fontFamily = 'monospace';
    this.toggleBtn.style.padding = '2px 6px';
    this.toggleBtn.style.cursor = 'pointer';
    this.toggleBtn.addEventListener('click', () => {
      this.setCollapsed(!this._collapsed);
    });

    this.summaryText = document.createElement('span');
    this.summaryText.textContent = ''; // e.g., "Done! Time: 5m 32s"

    this.summaryBar.appendChild(this.summaryText);
    this.summaryBar.appendChild(this.toggleBtn);

    // Details (live view)
    this.details = document.createElement('div');
    this.details.style.padding = '8px';
    this.details.style.fontFamily = 'monospace';
    this.details.style.whiteSpace = 'pre';

    this.title = document.createElement('div');
    this.title.style.fontWeight = '600';
    this.title.style.marginBottom = '4px';
    this.title.textContent = this.label;

    this.display = document.createElement('div');
    this.display.style.whiteSpace = 'pre';

    this.details.appendChild(this.title);
    this.details.appendChild(this.display);

    this.root.appendChild(this.summaryBar);
    this.root.appendChild(this.details);
    parent.appendChild(this.root);
  }

  setInitialTasks(count) {
    this.total = count;
    this.completed = 0;
    this.errors = 0;
    this.startTime = Date.now();
    this._ensureUiTimer();
    this._updateUI();
  }

  addTasks(count) {
    this.total += count;
    this._updateUI();
  }

  markResolved(ok) {
    this.completed += 1;
    if (!ok) this.errors += 1;
    this._updateUI();
  }

  getElapsedSeconds() {
    if (!this.startTime) return 0;
    const end = this.endTime ?? Date.now();
    return Math.max(0, (end - this.startTime) / 1000);
  }

  getSpeedRps() {
    const elapsed = this.getElapsedSeconds();
    if (elapsed <= 0) return 0;
    return this.completed / elapsed;
  }

  getRemainingSeconds() {
    const speed = this.getSpeedRps();
    if (speed <= 0) return Infinity;
    const remaining = Math.max(0, this.total - this.completed);
    return remaining / speed;
  }

  finalizeAndCollapse() {
    this.endTime = Date.now();
    this._stopUiTimer();
    const elapsed = this.getElapsedSeconds();
    if (this.errors > 0) {
      this.summaryText.textContent = `Completed with Errors: ${this._formatReadableTime(elapsed)}`;
      this.summaryBar.style.display = 'flex';
    } else {
      this.summaryText.textContent = `Done! Time: ${this._formatReadableTime(elapsed)}`;
      this.summaryBar.style.display = 'flex';
      this.setCollapsed(true);
    }
  }

  setCollapsed(collapsed) {
    this._collapsed = collapsed;
    this.details.style.display = collapsed ? 'none' : 'block';
    this.toggleBtn.textContent = collapsed ? '＋' : '－';
  }

  _ensureUiTimer() {
    if (this._uiTimer) return;
    this._uiTimer = setInterval(() => this._updateUI(), 1000);
  }

  _stopUiTimer() {
    if (this._uiTimer) {
      clearInterval(this._uiTimer);
      this._uiTimer = null;
    }
    this._updateUI();
  }

  _formatMMSS(seconds) {
    if (!isFinite(seconds)) return '--:--';
    const s = Math.max(0, Math.round(seconds));
    const mm = Math.floor(s / 60).toString();
    const ss = (s % 60).toString().padStart(2, '0');
    return `${mm}:${ss}`;
  }

  _formatReadableTime(seconds) {
    if (!isFinite(seconds)) return '--';
    const s = Math.max(0, Math.round(seconds));
    const m = Math.floor(s / 60);
    const r = s % 60;
    if (m === 0) return `${r}s`;
    return `${m}m ${r}s`;
  }

  _updateUI() {
    const progressLine = `Progress: ${this.completed}/${this.total}`;
    const elapsed = this.getElapsedSeconds();
    const remaining = this.getRemainingSeconds();
    const timeLine = `Time: ${this._formatMMSS(elapsed)} < ${this._formatMMSS(remaining)}`;
    const speed = this.getSpeedRps().toFixed(2);
    const speedLine = `Speed: ${speed}`;
    const errorsLine = `Errors: ${this.errors}`;
    this.display.innerText = `${progressLine}\n${timeLine}\n${speedLine}\n${errorsLine}`;
  }
}

/**
 * A rate-limited, retrying task queue for LLM completions.
 * - Supports enqueueing an array of prompts with a unified callback.
 * - Allows adding tasks while running.
 * - Enforces max concurrency + calls-per-second.
 * - Retries with backoff (up to MAX_RETRIES).
 *
 * Task callback signature:
 *    (result) => void
 * where result = {
 *   ok: boolean,
 *   response?: string,     // model output when ok === true
 *   error?: Error,         // final error when ok === false
 *   prompt: string,
 *   taskId: number,
 *   attempts: number
 * }
 */
class RequestQueue {
  constructor(client, progressMetrics) {
    this.client = client;
    this.metrics = progressMetrics;

    this.queue = []; // pending tasks
    this.active = 0; // in-flight tasks
    this.tokens = MAX_CALLS_PER_SEC;
    this._nextTaskId = 1;
    this._drainScheduled = false;

    // Refill tokens every second
    this._refillTimer = setInterval(() => {
      this.tokens = MAX_CALLS_PER_SEC;
      this._drain();
    }, 1000);

    this.inUse = true;
  }

  dispose() {
    if (this.inUse === false) {
      // Ensure queue can only be deleted once
      return;
    }

    this.inUse = false;
    clearInterval(this._refillTimer);
    this.metrics.finalizeAndCollapse();
  }

  /**
   * Enqueue a single prompt with its callback.
   * Returns a promise that resolves with the task result object.
   */
  enqueueTask(prompt, callback) {
    const taskId = this._nextTaskId++;
    const task = {
      taskId,
      prompt,
      callback,
      attempts: 0,
    };

    const p = new Promise((resolve) => {
      task._resolve = resolve; // resolve with result object on final success/failure
    });

    this.queue.push(task);
    this.metrics.addTasks(1);
    this._drain();
    return p;
  }

  /**
   * Enqueue multiple prompts with a shared callback.
   * Returns Promise<Array<result>>
   */
  enqueueAll(prompts, callback) {
    const promises = [];
    for (const prompt of prompts) {
      promises.push(this.enqueueTask(prompt, callback));
    }
    return Promise.all(promises);
  }

  /**
   * Internal: start as many tasks as allowed by concurrency and token limits.
   */
  _drain() {
    if (this._drainScheduled) return;
    this._drainScheduled = true;

    queueMicrotask(() => {
      this._drainScheduled = false;

      while (
        this.active < MAX_CONCURRENCY &&
        this.tokens > 0 &&
        this.queue.length > 0
        ) {
        const task = this.queue.shift();
        this.tokens -= 1;
        this.active += 1;
        void this._execute(task);
      }
    });
  }

  /**
   * Internal: execute a task with retry logic.
   */
  async _execute(task) {
    let finalResult = null;

    try {
      task.attempts += 1;

      // Call the client
      const response = await this.client.completion(task.prompt.system, task.prompt.user);

      finalResult = {
        ok: true,
        response,
        prompt: task.prompt,
        taskId: task.taskId,
        attempts: task.attempts,
      };

      // Call user callback safely
      try {
        task.callback && task.callback(finalResult);
      } catch (cbErr) {
        console.error('Callback error:', cbErr);
      }

      // Mark resolved
      this.metrics.markResolved(true);
      task._resolve(finalResult);

    } catch (err) {
      // Retry if allowed
      if (task.attempts < MAX_RETRIES) {
        const backoff = BASE_RETRY_DELAY_MS * Math.pow(2, task.attempts - 1);
        const jitter = Math.floor(Math.random() * 100);
        const delay = backoff + jitter;

        // Requeue after delay
        setTimeout(() => {
          // Put task back at the end of the queue; do not change total count
          this.queue.push(task);
          this._drain(); // try to resume when ready
        }, delay);
      } else {
        // Final failure
        finalResult = {
          ok: false,
          error: err,
          prompt: task.prompt,
          taskId: task.taskId,
          attempts: task.attempts,
        };

        try {
          task.callback && task.callback(finalResult);
        } catch (cbErr) {
          console.error('Callback error:', cbErr);
        }

        this.metrics.markResolved(false);
        errorHandler();
        task._resolve(finalResult);
      }
    } finally {
      this.active -= 1;
      // Opportunistically drain again to keep the pipeline full
      this._drain();
    }
  }
}

/* ================================================================================
    Domain Manager
    Provides domain-specific configurations and utilities across different sites
   ================================================================================*/

const DOMAIN_CONFIGS = {
  'file': {
    domainCode: '99',
    getSeriesId: (url) => {
      const parts = url.split('/');
      const filteredParts = parts.filter(part => part).slice(0, -1);
      return filteredParts.length > 0 ? filteredParts[filteredParts.length - 1] : '.';
    },
    extractText: () => {
      const paragraphMap = new Map();
      const pElements = Array.from(document.body.querySelectorAll('p'));
      let idCounter = 1;

      pElements.forEach(p => {
        const processedText = new TextPreProcessor(p.textContent)
          .getText();

        if (p.id) {
          paragraphMap.set(p.id, processedText);
        } else {
          const generatedId = `${SCRIPT_PREFIX}${idCounter++}`;
          p.id = generatedId;
          paragraphMap.set(generatedId, processedText);
        }
      });
      return paragraphMap;
    },
  },


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

          if (p.id && processedText) {
            paragraphMap.set(p.id, processedText);
          }
        });
      });
      return paragraphMap;
    },
  },


  'kakuyomu.jp': {
    domainCode: '2',
    getSeriesId: (url) => {
      const match = url.match(/\/works\/(\d+)/);
      return match ? match[1] : null;
    },
    extractText: () => {
      const textContainers = document.querySelectorAll('.js-episode-body');
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

          if (p.id && !p.classList.contains('blank') && processedText) {
            paragraphMap.set(p.id, processedText);
          }
        });
      });
      return paragraphMap;
    },
  },
};

class DomainManager {
  constructor() {
    this.currentConfig = this.detectDomain();
  }

  detectDomain() {
    if (window.location.protocol === 'file:') {
      return {
        ...DOMAIN_CONFIGS['file'],
        domain: 'file',
      };
    }

    const hostname = window.location.hostname;
    for (const [domain, config] of Object.entries(DOMAIN_CONFIGS)) {
      if (hostname.endsWith(domain)) {
        return {
          ...config,
          domain,
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

function extractTextFromWebpage(domainManager) {
  const config = domainManager.currentConfig;
  return config.extractText();
}

/* ================================================================================
   LLM response processing
   JSON / XML extraction and validation of LLM responses
   ================================================================================*/

/**
 * Extracts the content enclosed within a specified XML/HTML tag from a given string.
 * If the closing tag is missing, it attempts to return the content after the opening tag.
 * If neither the opening nor closing tag is found, it provides a fallback value.
 *
 * @param {string} str - The input string containing the XML/HTML content.
 * @param {string} tag - The name of the tag to extract content from.
 * @returns {string} - The extracted content, or a fallback value ('###') if the tag is not found.
 */
function extractTextFromTag(str, tag) {
  // Escape special regex characters in tag
  const escapedTag = tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

  const openingTag = `<${tag}>`;
  const closingTag = `</${tag}>`;

  // Try to extract all balanced tag pairs
  const balancedRegex = new RegExp(`<${escapedTag}>(.*?)<\\/${escapedTag}>`, 'gs');
  const matches = [...str.matchAll(balancedRegex)];

  if (matches.length > 0) {
    return matches.map(match => match[1].trim()).join('\n');
  }

  // Count opening and closing tags
  const openingMatches = str.match(new RegExp(`<${escapedTag}>`, 'g'));
  const closingMatches = str.match(new RegExp(`<\\/${escapedTag}>`, 'g'));
  const openingCount = openingMatches ? openingMatches.length : 0;
  const closingCount = closingMatches ? closingMatches.length : 0;

  // Helper to find all indices of a substring
  const findAllIndices = (text, substring) => {
    const indices = [];
    let index = 0;
    while ((index = text.indexOf(substring, index)) !== -1) {
      indices.push(index);
      index += substring.length;
    }
    return indices;
  };

  // Case 1: Missing first opening tag (one more closing than opening)
  if (closingCount === openingCount + 1 && closingCount > 0) {
    console.warn(`Warning: Missing first opening tag <${tag}>. Attempting recovery.\n${str}`);

    const openingIndices = findAllIndices(str, openingTag);
    const closingIndices = findAllIndices(str, closingTag);
    const extractedTexts = [];

    // First segment: from start to first closing tag
    extractedTexts.push(str.slice(0, closingIndices[0]).trim());

    // Remaining segments: balanced pairs
    for (let i = 0; i < openingIndices.length; i++) {
      const start = openingIndices[i] + openingTag.length;
      const end = closingIndices[i + 1];
      extractedTexts.push(str.slice(start, end).trim());
    }

    return extractedTexts.join('\n');
  }

  // Case 2: Missing last closing tag (one more opening than closing)
  if (openingCount === closingCount + 1 && openingCount > 0) {
    console.warn(`Warning: Missing last closing tag </${tag}>. Attempting recovery.\n${str}`);

    const openingIndices = findAllIndices(str, openingTag);
    const closingIndices = findAllIndices(str, closingTag);
    const extractedTexts = [];

    // All but last: balanced pairs
    for (let i = 0; i < closingIndices.length; i++) {
      const start = openingIndices[i] + openingTag.length;
      const end = closingIndices[i];
      extractedTexts.push(str.slice(start, end).trim());
    }

    // Last segment: from last opening to end
    const lastStart = openingIndices[openingIndices.length - 1] + openingTag.length;
    extractedTexts.push(str.slice(lastStart).trim());

    return extractedTexts.join('\n');
  }

  // Case 3: No tags found or too broken
  if (openingCount === 0 && closingCount === 0) {
    console.warn(`Warning: No tags <${tag}> found.\n${str}`);
  } else {
    console.warn(`Warning: Tags too malformed to recover. Opening: ${openingCount}, Closing: ${closingCount}.\n${str}`);
  }

  return '###';
}

/**
 * Parses JSON data from the output of an LLM (Large Language Model).
 * This function attempts to extract and parse JSON from various formats
 * and structures commonly found in LLM outputs, including code fences,
 * comments, and unbalanced JSON structures.
 *
 * @param {string} llmOutput - The raw output string from the LLM.
 * @returns {object} - The parsed JSON object.
 * @throws {Error} - Throws an error if no valid JSON is found.
 */
function parseJSONFromLLM(llmOutput) {
  // Strategy 1: Look for ```json fence (expected format)
  let match = llmOutput.match(/```json\s*\n([\s\S]*?)\n?```/);
  if (match) {
    try {
      return JSON.parse(match[1].trim());
    } catch (e) {
      // Malformed, continue to other strategies
    }
  }

  // Strategy 2: Look for any code fence (``` with any or no language identifier)
  const fenceMatches = [...llmOutput.matchAll(/```\w*\s*\n([\s\S]*?)\n?(?:```|$)/g)];
  for (const fenceMatch of fenceMatches) {
    const content = fenceMatch[1].trim();
    // Try parsing as-is
    try {
      return JSON.parse(content);
    } catch (e) {
      // Try removing JavaScript-style comments
      try {
        const cleaned = content
          .replace(/\/\/.*$/gm, '') // Remove line comments
          .replace(/\/\*[\s\S]*?\*\//g, ''); // Remove block comments
        return JSON.parse(cleaned);
      } catch (e2) {
        // Try next fence
      }
    }
  }

  const extractBalancedJSON = (text) => {
    // Try both object and array formats
    for (const [startChar, endChar] of [['{', '}'], ['[', ']']]) {
      const startIndex = text.indexOf(startChar);
      if (startIndex === -1) continue;

      let depth = 0;
      let inString = false;
      let escapeNext = false;

      for (let i = startIndex; i < text.length; i++) {
        const char = text[i];

        if (escapeNext) {
          escapeNext = false;
          continue;
        }

        if (char === '\\') {
          escapeNext = true;
          continue;
        }

        if (char === '"') {
          inString = !inString;
          continue;
        }

        if (!inString) {
          if (char === startChar) depth++;
          else if (char === endChar) {
            depth--;
            if (depth === 0) {
              return text.substring(startIndex, i + 1);
            }
          }
        }
      }
    }

    return null;
  }

  // Strategy 3: Extract balanced JSON object or array from raw text
  const extracted = extractBalancedJSON(llmOutput);
  if (extracted) {
    try {
      return JSON.parse(extracted);
    } catch (e) {
      // Try removing comments from extracted JSON
      try {
        const cleaned = extracted
          .replace(/\/\/.*$/gm, '')
          .replace(/\/\*[\s\S]*?\*\//g, '');
        return JSON.parse(cleaned);
      } catch (e2) {
        // Failed
      }
    }
  }

  console.error(`No valid JSON found in LLM output. Raw response:\n\n${llmOutput}`);

  return {};
}

/* ================================================================================
   Persistant storage
   Code for reading / saving data to disk via GM get/set
   ================================================================================*/

function loadDictionary(domainManager) {
  const seriesId = domainManager.getCurrentSeriesId();

  if (seriesId) {
    const dictionaryKey = domainManager.getDictionaryKey(seriesId);
    return GM_getValue(dictionaryKey, { entries: [] });
  } else {
    console.error('Failed to extract series ID from URL');
    return { entries: [] };
  }
}

function saveDictionary(domainManager, dictionary) {
  const seriesId = domainManager.getCurrentSeriesId();

  if (seriesId) {
    const dictionaryKey = domainManager.getDictionaryKey(seriesId);
    GM_setValue(dictionaryKey, dictionary);
  } else {
    console.error('Failed to extract series ID from URL');
  }
}

function loadApiKeys() {
  function areSetsEqual(setA, setB) {
    if (setA.size !== setB.size) return false;
    for (const item of setA) {
      if (!setB.has(item)) return false;
    }
    return true;
  }

  const savedKeys = GM_getValue("apiKeys", {});

  // Validate that both configs have the same provider sets
  const apiConfigProviders = new Set(Object.keys(PROVIDER_API_CONFIG));
  const modelConfigProviders = new Set(Object.keys(PROVIDER_CONFIGS));
  if (!areSetsEqual(apiConfigProviders, modelConfigProviders)) {
    console.warn('Provider mismatch detected between API config and model config');
  }

  // Populate modelsList with avalible models for use (those with API keys set)
  modelsList.length = 0;

  Object.keys(PROVIDER_API_CONFIG).forEach(provider => {
    if (savedKeys[provider]) {
      PROVIDER_API_CONFIG[provider].apiKey = savedKeys[provider];
      if (PROVIDER_CONFIGS[provider] && PROVIDER_CONFIGS[provider].models) {
        PROVIDER_CONFIGS[provider].models.forEach(model => {
          modelsList.push(model.id);
        });
      }
    }
  });

  return savedKeys;
}

function saveApiKeys(keys) {
  // Filter out empty strings
  const sanitizedKeys = {};
  Object.keys(keys).forEach(provider => {
    if (typeof keys[provider] === 'string' && keys[provider].trim() !== '') {
      sanitizedKeys[provider] = keys[provider].trim();
    }
  });

  GM_setValue("apiKeys", sanitizedKeys);

  // Also update runtime config
  Object.keys(PROVIDER_API_CONFIG).forEach(provider => {
    PROVIDER_API_CONFIG[provider].apiKey = sanitizedKeys[provider] || null;
  });
}

/* ================================================================================
   Text Pre/Post Processing
   Extraction of text from web page, and updating it with translation
   ================================================================================*/

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
    // const pattern = new RegExp('[　◇◆♦＊_＿─\*\\♦︎]+', 'g');
    // this.text = this.text.replace(pattern, '');
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

function textPostProcess(str) {
  let result = str.trim();

  // Transform to smart quotes "..." -> “...”
  const prefixes = ["\"", "＂", "“"];
  const maxOffset = 3;

  if (
    prefixes.some(prefix =>
      Array.from({ length: maxOffset + 1 }, (_, i) =>
        str.startsWith(prefix, i),
      ).some(Boolean),
    )) {
    result = result.replace(/^"/, '“').replace(/"([^"]*)$/, '”$1');
  }

  return result;
}

function updateParagraphContent(id, newContent) {
  const paragraph = document.getElementById(id);
  if (paragraph) {
    paragraph.innerHTML = newContent;
    paragraph.classList.add(`${SCRIPT_PREFIX}text`);
  }
}

/* ================================================================================
   Glossary
   Generate and updates the glossary / dictionary
   ================================================================================*/

/**
 * Main orchestrator for glossary generation and updates
 */
class GlossaryManager {
  constructor(uiConfig) {
    this.stage1 = new Stage1Generator(uiConfig);
    this.stage2 = new Stage2Updater(uiConfig);
  }

  async generateAndUpdateDictionary(dictionary, texts) {
    // Stage 1: Generate new glossary entries from text chunks
    const newEntries = await this.stage1.generate(texts);
    console.log(`Stage 1 complete: generated ${newEntries.length} new entries`);
    // this.queue_stage1.dispose();

    // Stage 2: Merge new entries with existing dictionary
    const updatedDictionary = await this.stage2.update(dictionary, newEntries);
    console.log(`Stage 2 complete: dictionary now has ${updatedDictionary.entries.length} entries`);
    // this.queue_stage2.dispose();

    return updatedDictionary;
  }
}

/**
 * Stage 1: Generate glossary entries from text chunks
 * Does NOT see existing dictionary
 */
class Stage1Generator {
  constructor(uiConfig) {
    this.uiConfig = uiConfig;
    this._queue = null;
  }

  get queue() {
    if (!this._queue) {
      this._queue = createStageQueue(this.uiConfig.stage1, 'Glossary Generation');
    }
    return this._queue;
  }

  async generate(texts) {
    // Chunk texts into blocks to fit in LLM context window
    const chunks = this._chunkTexts(texts);

    // Create prompts for each chunk
    const prompts = chunks.map(chunk => this._createPrompt(chunk));

    // Run inference in parallel
    const results = await this.queue.enqueueAll(
      prompts,
      (result) => {
        if (!result.ok) {
          console.error(`[Stage 1] Task ${result.taskId} failed:`, result.error);
        }
      },
    );

    this.queue.dispose();

    // Consolidate all responses into single entry array
    return this._consolidateResponses(results);
  }

  _chunkTexts(texts) {
    const chunks = [];
    const maxChunkSize = GLOSSARY_CHUNK_SIZE;
    let currentChunk = '';

    for (const [_, text] of texts) {
      if (currentChunk.length + text.length > maxChunkSize && currentChunk.length > 0) {
        chunks.push(currentChunk);
        currentChunk = text;
      } else {
        currentChunk += (currentChunk ? '\n' : '') + text;
      }
    }

    if (currentChunk) {
      chunks.push(currentChunk);
    }

    return chunks;
  }

  _createPrompt(chunk) {
    const system_prompt = `
You are generating entries to a multi-key dictionary that will be used as the knowledge base for a RAG pipeline in a Japanese to English LLM translation task. This metadata part will be used to help ensure consistency in translation of names and proper nouns across multiple API calls. In the RAG pipeline, the text to be translated will be scanned and presence of any of keys will result in the content of "value" to be included in the LLM context.

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

3. Entry Selection Criteria:
   - Focus on character names, location names, proper nouns, and special terms
   - Exclude common nouns or terms
   - Skip terms if unclear or lacking sufficient context

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
      "value": "[character] Name: John Doe (名無しの権兵衛) | Gender: Male | Nickname: Nanashi (ななし)"
    },
    {
      "keys": ["アメリカ合衆国", "アメリカ"],
      "value": "[location] Name: United States (アメリカ)"
    }
  ]
}

Output only the JSON, without any commentary.

You will be provided the raw text delimited with <text> XML tags.
`.trim();

    return {
      system: system_prompt,
      user: `<text>\n${chunk}\n</text>`,
    };
  }

  _consolidateResponses(results) {
    const allEntries = [];

    for (const result of results) {
      if (result.ok) {
        try {
          const parsed = parseJSONFromLLM(result.response);
          if (validateStage1Response(parsed)) {
            allEntries.push(...parsed.entries);
          }
        } catch (err) {
          console.error('Failed to parse Stage 1 response:', err);
        }
      }
    }

    return allEntries;
  }
}

/**
 * Validate Stage 1 response structure
 */
function validateStage1Response(obj) {
  if (!obj || !Array.isArray(obj.entries)) {
    return false;
  }

  const valuePattern = /^\[.*] .*$/;

  return obj.entries.every(entry => {
    if (!entry || typeof entry !== 'object') {
      return false;
    }

    const { keys, value } = entry;

    if (!Array.isArray(keys) || keys.length === 0) {
      return false;
    }

    const allKeysAreStrings = keys.every(key => typeof key === 'string');
    const allKeysAreUnique = new Set(keys).size === keys.length;

    if (!allKeysAreStrings || !allKeysAreUnique) {
      return false;
    }

    if (typeof value !== 'string' || !valuePattern.test(value)) {
      return false;
    }

    // If we reach here, object should be in expected shape
    return true;
  });
}

// Minimal async mutex to separate pending entries selection / schedulding from dictionary update / mutation
// Technically not requried since the 2 critical sections are currently not async (find new pending), (execute update actions)
class AsyncMutex {
  constructor() {
    this._locked = false;
    this._waiters = [];
  }

  acquire() {
    return new Promise((resolve) => {
      const take = () => {
        this._locked = true;
        resolve(this._release.bind(this));
      };
      if (!this._locked) take();
      else this._waiters.push(take);
    });
  }

  _release() {
    this._locked = false;
    const next = this._waiters.shift();
    if (next) next();
  }

  async runExclusive(fn) {
    const release = await this.acquire();
    try {
      return await fn();
    } finally {
      release();
    }
  }
}

/**
 * Stage 2: Merge new entries with existing dictionary (parallel, lock-safe)
 * - 1) Select the largest possible batch of non-conflicting new entries.
 *   - Zero-conflict entries are added immediately (no LLM).
 *   - Conflicting entries are sent to LLM (in parallel via RequestQueue).
 * - 2) Send LLM requests for all selected conflicting entries.
 * - 3) When a request resolves, apply changes under a mutex, unlock keys, then try to schedule more.
 *
 * Locking model:
 * - For each new entry E with conflicts, lock set L(E) = keys(E) ∪ ⋃ keys(conflict entries of E).
 * - Two entries can be processed concurrently iff their lock sets are disjoint.
 * - Unlock uses the original L(E) computed at scheduling time.
 */
class Stage2Updater {
  constructor(uiConfig) {
    this.uiConfig = uiConfig;
    this._queue = null;
    this.nextId = 1;
    this._mutex = new AsyncMutex();
    this._usedKeys = new Set(); // union of lock sets of all in-flight tasks
  }

  get queue() {
    if (!this._queue) {
      this._queue = createStageQueue(this.uiConfig.stage2, 'Glossary Update');
    }
    return this._queue;
  }

  async update(existingDict, newEntries) {
    const workingDict = this._cloneDictionary(existingDict);
    this.nextId = this._getMaxId(workingDict) + 1;

    // Pending work (preserve relative order to reduce starvation)
    const pending = newEntries.map((entry, idx) => ({ idx, entry }));

    // In-flight requests: [{ id, p }]
    const inFlight = [];
    let seq = 1;

    // Find a maximal batch of entries whose lock sets do not intersect _usedKeys.
    const scheduleAvailable = async () => {
      await this._mutex.runExclusive(async () => {
        // Try to schedule until either no candidate fits
        for (let i = 0; i < pending.length;) {
          const candidate = pending[i];

          const { conflicts, lockKeys } = this._computeLocks(workingDict, candidate.entry);

          // No conflict: add immediately, no LLM needed
          if (conflicts.length === 0) {
            this._addEntry(workingDict, candidate.entry);
            pending.splice(i, 1);
            // Don't increment i since we removed current index
            continue;
          }

          // Has conflicts: check if we can schedule LLM request
          if (this._intersects(lockKeys, this._usedKeys)) {
            i += 1; // cannot schedule this one now; try next
            continue;
          }

          // Lock keys for this task
          for (const k of lockKeys) this._usedKeys.add(k);

          // Build prompt and enqueue (Part 2)
          const prompt = this._createConflictPrompt(conflicts, candidate.entry);

          const basePromise = this.queue.enqueueTask(prompt, () => {
          });

          // Attach metadata and a local sequence id so we can identify which promise completed.
          const meta = {
            entry: candidate.entry,
            conflicts,
            lockKeys: Array.from(lockKeys),
          };
          const id = seq++;

          const wrapped = basePromise.then((result) => ({ result, meta, id }));

          inFlight.push({ id, p: wrapped });

          // Remove from pending once scheduled
          pending.splice(i, 1);
          // Do not increment i here, as we removed the current index
        }
      });
    };

    // Kick off the first batch
    await scheduleAvailable();

    // Process completions as they arrive
    while (pending.length > 0 || inFlight.length > 0) {
      if (inFlight.length === 0) {
        // No work in-flight; try scheduling again
        await scheduleAvailable();
        if (inFlight.length === 0) break; // nothing schedulable (should not happen with empty usedKeys)
      }

      // Wait for one completion
      const { result, meta, id } = await Promise.race(inFlight.map((it) => it.p));

      // Apply LLM result under mutex, then unlock and attempt more scheduling
      await this._mutex.runExclusive(async () => {
        if (result.ok) {
          try {
            const parsed = parseJSONFromLLM(result.response);
            const actions = this._normalizeActions(parsed);
            const validationError = this._validateActions(actions, meta.conflicts);
            if (!validationError) {
              this._executeActions(workingDict, actions, meta.entry);
            } else {
              console.error('Action validation failed:', validationError);
              // Treat as no-op
            }
          } catch (err) {
            console.error('Failed to parse/execute actions:', err);
            // Treat as no-op
          }
        } else {
          console.warn('Glossary update LLM call failed.');
          // Treat as no-op
        }

        // Unlock original lock keys
        for (const k of meta.lockKeys) this._usedKeys.delete(k);

        // Remove this promise from in-flight
        const idx = inFlight.findIndex((it) => it.id === id);
        if (idx !== -1) inFlight.splice(idx, 1);
      });

      // See if freeing these keys opens new scheduling opportunities
      await scheduleAvailable();
    }

    this.queue.dispose();
    return workingDict;
  }

  // Compute conflicts (existing entries that intersect keys) and the lock set
  _computeLocks(workingDict, newEntry) {
    const conflicts = this._findConflicts(workingDict, newEntry);
    const lockKeys = new Set(newEntry.keys);
    for (const c of conflicts) {
      for (const k of c.keys) lockKeys.add(k);
    }
    return { conflicts, lockKeys };
  }

  _intersects(keysSet, usedKeys) {
    for (const k of keysSet) {
      if (usedKeys.has(k)) return true;
    }
    return false;
  }

  _findConflicts(workingDict, newEntry) {
    const newKeys = new Set(newEntry.keys);
    const conflicts = [];

    for (const existingEntry of workingDict.entries) {
      // Check for key intersection
      const hasIntersection = existingEntry.keys.some(key => newKeys.has(key));
      if (hasIntersection) {
        conflicts.push(existingEntry);
      }
    }

    return conflicts;
  }

  _addEntry(workingDict, entry) {
    const newEntry = {
      id: this.nextId++,
      keys: [...entry.keys],
      value: entry.value,
    };
    workingDict.entries.push(newEntry);
  }

  _createConflictPrompt(conflicts, newEntry) {
    const existingDict = {
      entries: conflicts,
    };

    const newUpdates = {
      entries: [newEntry],
    };

    const systemPrompt = `
You are in charge of merging and updating the glossary or dictionary for a translation system using a RAG pipeline.

Goal
- Maintain consistency across chapters by merging proposed glossary entries into an existing dictionary subset.
- Prefer existing translations and only update when it improves translation quality without breaking consistency.
- Output only JSON actions that the caller will execute; do not include any explanation or text outside the JSON.

Context and Inputs
- You will receive two sections:
  <existing_dictionary> { "entries": [ { "id": number, "keys": string[], "value": string }, ... ] } </existing_dictionary>
  <new_updates> { "entries": [ { "keys": string[], "value": string }, ... ] } </new_updates>
- The existing_dictionary contains only the candidate entries you are allowed to modify (a subset of the full dictionary).
- The new_updates are suggestions generated without seeing the dictionary; they often duplicate existing content.
- Only the "value" field will be visible to the later translation model. Make "value" concise and directly useful for translation.

Output Format (strict)
- Respond with ONLY one of the following:
  - A single JSON object: { "action": "none" }
  - OR a JSON array of action objects: [ { "action": "...", ... }, { ... } ]
- Allowed actions:
  - { "action": "none" }
  - { "action": "add_entry" }
  - { "action": "delete", "id": number }
  - { "action": "update", "id": number, "data": string }       // replace the entire 'value' of target id
  - { "action": "add_key", "id": number, "data": string[] }    // add these keys to target id
  - { "action": "del_key", "id": number, "data": string[] }    // remove these keys from target id
- Constraints:
  - IDs must be taken only from <existing_dictionary>. Never invent IDs.
  - add_entry has no id or data; the caller will append the new entry as-provided in <new_updates>. Downsteam client will take care of the id value.
  - For update, data must be a single string (the replacement value).
  - For add_key/del_key, data must be an array of strings.
  - Output must be valid JSON. No code fences, comments, or extra keys.

Canonicalization & Style Rules
- Keys:
  - keys must be raw Japanese strings that could appear in source text (kanji/kana). Do NOT add English or romaji to keys.
  - Prefer to keep previously-seen variants (kanji, kana, nicknames). Do not remove earlier keys unless they are clearly wrong or non-Japanese.
- Value:
  - Include both English and Japanese in the form: English (日本語)
  - Keep information directly useful for translation only. Trim verbose or narrative text.
  - Preserve the leading category tag if present, e.g. [character], [term], [location].
  - Recommended format and order:
    [category] Name: EN (JP) | Gender: ... | Title: ... | Nickname: EN (JP) | Note: ...
  - Keep “Note” short and only when it affects translation (aliases, romanization choice, honorific behavior).
- Consistency:
  - If the existing dictionary translates a Japanese term one way, keep that translation. Do NOT switch to synonyms (e.g., keep “Messiah” over a new “Saviour” proposal).
  - Prefer no-op if the new suggestion conflicts only by English synonym choice or superficial formatting.
  - Only refine an existing value when it removes useless text or adds necessary alias/role info that helps translation.

Decision Procedure (high level)
1) Identify the canonical entry to keep when multiple existing entries refer to the same concept:
   - Prefer the entry with more complete, translation-relevant info or the one already using a stable name choice.
   - Plan to update that one, add missing keys to it, and delete redundant duplicates.

2) Compare new_updates against existing_dictionary:
   - If new’s EN(JP) pairing matches an existing entry (same JP, same or acceptable EN), this is a no-op unless keys from new are useful variants → use add_key on the canonical id.
   - If new’s “value” is verbose or contains useless description, and it matches an existing concept, you may update the canonical id to a trimmed value that still includes all essential EN (JP) pairs.
   - If new is truly novel relative to all provided existing entries (different concept), return { "action": "add_entry" }.

3) Keys management:
   - Add kana/kanji/nickname variants using add_key on the canonical entry.
   - Remove keys that are clearly non-Japanese (pure ASCII English words or romanization) using del_key.
   - Avoid removing previously valid Japanese variants.

4) De-duplication:
   - If two or more existing entries are duplicates/aliases of the same concept, merge:
     - update the canonical entry’s value (if needed) to include the most useful, concise info.
     - add_key to include all relevant JP variants (kanji/kana/nicknames).
     - delete the redundant entries.

5) Minimality:
   - Prefer { "action": "none" } when the state is already correct.
   - Prefer the smallest set of actions to reach the improved state.

Safety
- Only touch IDs shown in <existing_dictionary>. Never reference unseen IDs.
- If uncertain, choose { "action": "none" }.
- Do not invent new facts or English names that do not appear in inputs.

Examples

Example A: No-op due to synonym; keep existing translation
Input:
  existing:
    id: 1, value: "[term] Name: Messiah (救世主)"
  new:
    value: "[term] Name: Saviour (救世主)"
Output:
  { "action": "none" }

Example B: Add a kana reading as key
Input:
  existing:
    id: 42, keys: ["名無しの権兵衛"], value: "[character] Name: John Doe (名無しの権兵衛)"
  new:
    keys: ["ななしのごんべい"], value: "[character] Name: John Doe (名無しの権兵衛)"
Output:
  [{ "action": "add_key", "id": 42, "data": ["ななしのごんべい"] }]

Example C: Trim verbose value (still same concept)
Input:
  existing:
    id: 7, value: "[character] Name: Sakura Miko (さくらみこ) | Description: A long and unnecessary blurb..."
  new:
    value: "[character] Name: Sakura Miko (さくらみこ)"
Output:
  [{ "action": "update", "id": 7, "data": "[character] Name: Sakura Miko (さくらみこ)" }]

Example D: Combine duplicates and keep one canonical entry
Input:
  existing:
    id: 3, keys: ["東雲","しののめ"], value: "[character] Name: Shinonome (東雲) | Gender: Female"
    id: 5, keys: ["氷姫"], value: "[character] Name: Ice Princess (氷姫) | Gender: Female | Note: A nickname for Shinonome."
  new:
    keys: ["東雲"], value: "[character] Name: Shinonome (東雲) | Gender: Female"
Output:
  [
    { "action": "update", "id": 3, "data": "[character] Name: Shinonome (東雲) | Gender: Female | Nickname: Ice Princess (氷姫)" },
    { "action": "add_key", "id": 3, "data": ["氷姫"] },
    { "action": "delete", "id": 5 }
  ]

Example E: Remove non-Japanese key
Input:
  existing:
    id: 9, keys: ["天照", "Amaterasu"], value: "[term] Name: Amaterasu (天照)"
  new:
    keys: ["天照"], value: "[term] Name: Amaterasu (天照)"
Output:
  [{ "action": "del_key", "id": 9, "data": ["Amaterasu"] }]

Example F: Truly novel entry (no true conflict)
Input:
  existing:
    id: 17, keys: ["京都"], value: "[location] Name: Kyoto (京都)"
  new:
    keys: ["大阪"], value: "[location] Name: Osaka (大阪)"
Output:
  [{ "action": "add_entry" }]

Final Reminder
- Return only valid JSON with the allowed actions.
- Use the minimal set of actions necessary.
- When in doubt, prefer { "action": "none" }.
`.trim();
    const userPrompt = `
<existing_dictionary>
${JSON.stringify(existingDict, null, 2)}
</existing_dictionary>

<new_updates>
${JSON.stringify(newUpdates, null, 2)}
</new_updates>
`.trim();

    return {
      system: systemPrompt,
      user: userPrompt,
    };
  }

  /**
   * Normalize actions: handle both single object and array
   * Input: { action: "none" } OR [{ action: "delete", id: 7 }, ...]
   * Output: Always an array
   */
  _normalizeActions(parsed) {
    if (Array.isArray(parsed)) {
      return parsed;
    }
    // Single object (e.g., { action: "none" })
    return [parsed];
  }

  /**
   * Validate actions structure and content
   * Returns: null if valid, error string if invalid
   */
  _validateActions(actions, conflicts) {
    const validActions = new Set(['update', 'delete', 'add_key', 'del_key', 'add_entry', 'none']);
    const conflictIds = new Set(conflicts.map(c => c.id));

    for (let i = 0; i < actions.length; i++) {
      const action = actions[i];

      // Check action is valid
      if (!action.action || !validActions.has(action.action)) {
        return `Action ${i}: invalid action type '${action.action}'`;
      }

      // Validate based on action type
      switch (action.action) {
        case 'none':
          // No additional validation needed
          break;

        case 'add_entry':
          // No id or data required
          if (action.id !== undefined || action.data !== undefined) {
            return `Action ${i} (add_entry): should not have 'id' or 'data' fields`;
          }
          break;

        case 'delete':
          // Requires only id
          if (!action.id) {
            return `Action ${i} (delete): missing 'id' field`;
          }
          if (!conflictIds.has(action.id)) {
            return `Action ${i} (delete): id ${action.id} not in conflict set`;
          }
          if (action.data !== undefined) {
            return `Action ${i} (delete): should not have 'data' field`;
          }
          break;

        case 'update':
          // Requires id and data (string)
          if (!action.id) {
            return `Action ${i} (update): missing 'id' field`;
          }
          if (!conflictIds.has(action.id)) {
            return `Action ${i} (update): id ${action.id} not in conflict set`;
          }
          if (typeof action.data !== 'string') {
            return `Action ${i} (update): 'data' must be a string`;
          }
          break;

        case 'add_key':
          // Requires id and data (array of strings)
          if (!action.id) {
            return `Action ${i} (add_key): missing 'id' field`;
          }
          if (!conflictIds.has(action.id)) {
            return `Action ${i} (add_key): id ${action.id} not in conflict set`;
          }
          if (!Array.isArray(action.data)) {
            return `Action ${i} (add_key): 'data' must be an array`;
          }
          if (!action.data.every(k => typeof k === 'string')) {
            return `Action ${i} (add_key): all keys in 'data' must be strings`;
          }
          break;

        case 'del_key':
          // Requires id and data (array of strings)
          if (!action.id) {
            return `Action ${i} (del_key): missing 'id' field`;
          }
          if (!conflictIds.has(action.id)) {
            return `Action ${i} (del_key): id ${action.id} not in conflict set`;
          }
          if (!Array.isArray(action.data)) {
            return `Action ${i} (del_key): 'data' must be an array`;
          }
          if (!action.data.every(k => typeof k === 'string')) {
            return `Action ${i} (del_key): all keys in 'data' must be strings`;
          }
          break;
      }
    }

    return null; // Valid
  }

  _executeActions(workingDict, actions, newEntry) {
    for (const action of actions) {
      try {
        this._executeAction(workingDict, action, newEntry);
      } catch (err) {
        console.error('Failed to execute action:', action, err);
      }
    }
  }

  _executeAction(workingDict, action, newEntry) {
    switch (action.action) {
      case 'update':
        this._actionUpdate(workingDict, action.id, action.data);
        break;
      case 'delete':
        this._actionDelete(workingDict, action.id);
        break;
      case 'add_key':
        this._actionAddKey(workingDict, action.id, action.data);
        break;
      case 'del_key':
        this._actionDelKey(workingDict, action.id, action.data);
        break;
      case 'add_entry':
        this._addEntry(workingDict, newEntry);
        break;
      case 'none':
        // No-op
        break;
      default:
        console.warn(`Unknown action type: ${action.action}`);
    }
  }

  _actionUpdate(workingDict, id, newValue) {
    const entry = workingDict.entries.find(e => e.id === id);
    if (entry) {
      entry.value = newValue;
    } else {
      console.warn(`Update action: entry ${id} not found`);
    }
  }

  _actionDelete(workingDict, id) {
    const index = workingDict.entries.findIndex(e => e.id === id);
    if (index !== -1) {
      workingDict.entries.splice(index, 1);
    } else {
      console.warn(`Delete action: entry ${id} not found`);
    }
  }

  _actionAddKey(workingDict, id, keys) {
    const entry = workingDict.entries.find(e => e.id === id);
    if (entry) {
      entry.keys.push(...keys);
      // Remove duplicates
      entry.keys = [...new Set(entry.keys)];
    } else {
      console.warn(`Add key action: entry ${id} not found`);
    }
  }

  _actionDelKey(workingDict, id, keys) {
    const entry = workingDict.entries.find(e => e.id === id);
    if (entry) {
      const keysToRemove = new Set(keys);
      entry.keys = entry.keys.filter(k => !keysToRemove.has(k));
    } else {
      console.warn(`Delete key action: entry ${id} not found`);
    }
  }

  _cloneDictionary(dict) {
    return JSON.parse(JSON.stringify(dict));
  }

  _getMaxId(dict) {
    if (!dict.entries || dict.entries.length === 0) {
      return 0;
    }
    return Math.max(...dict.entries.map(e => e.id));
  }
}

/* ================================================================================
   Translation
   Prompts for transatlion, for different methods. Stage3a and stage3b logic
   ================================================================================*/

/**
 * A stateful prompt builder for a single translation session.
 * It is initialized with the UI configuration and dictionary,
 * and then generates specific prompts for each task.
 */
class PromptManager {
  /**
   * @param {object} uiConfig - The configuration object from the UI.
   * @param {object} dictionary - The glossary for the current series.
   */
  constructor(uiConfig, dictionary) {
    this.uiConfig = uiConfig;
    this.dictionary = dictionary;
    this._buildBasePrompts(); // "Bake" the static parts of the prompts upon creation.
  }

  _buildBasePrompts() {
    // --- For Translation ---
    this.baseTranslationSystem = `
You are a highly skilled Japanese to English literature translator, tasked with translating text from Japanese to English. Aim to maintain the original tone, prose, nuance, and character voices of the source text as closely as possible.
Do not under any circumstances localize anything by changing the original meaning or tone, stick strictly to translating the original tone, prose and language as closely as possible to the original text.
`.trim();

    const instructions = {
      narrative: undefined,
      nameOrder: undefined,
      honorifics: undefined,
    }

    switch (this.uiConfig.narrative) {
      case "auto":
        instructions.narrative = "Determine which narrative voice (first person, third person) the text is best translated as.";
        break;
      case "first":
        instructions.narrative = "For non-dialogue text (narration, description), default to using a first-person narrative voice unless the original raw text strongly indicates a different narrative style.";
        break;
      case "third":
        instructions.narrative = "For non-dialogue text (narration, description), default to using a third-person narrative voice unless the original raw text strongly indicates a different narrative style.";
        break;
      default:
        instructions.narrative = "";
        break;
    }

    switch (this.uiConfig.honorifics) {
      case "preserve":
        instructions.honorifics = `
When translating names, preserve honorifics if they are present in the original text.
<example>
'花子さん' -> 'Hanako-san'
'花子様' -> 'Hanako-sama'
</example>
`.trim();
        break;
      case "nil":
        instructions.honorifics = `
When translating names, drop common honorifics. You may choose to change them to a suitable English equivalant depending on the context.
<example>
'花子さん' -> 'Hanako'
'花子殿' -> 'Miss Hanako'
</example>
`.trim();
        break;
    }

    switch (this.uiConfig.nameOrder) {
      case 'jp':
        instructions.nameOrder = `
Maintain the same same Japanese name ordering (LastName-FirstName) in your English translation.
<example>
'山田太郎' -> 'Yamada Taro'
'琴 紗月' -> 'Koto Satsuki'
</example>
`.trim();
        break;
      case 'en':
        instructions.nameOrder = `
Use English name ordering (FirstName-LastName) in your translation.
<example>
'山田太郎' -> 'Taro Yamada'
'琴 紗月' -> 'Satsuki Koto'
</example>
`.trim();
        break;
    }

    this.baseTranslationDev = `
<instructions>
### Guiding Principles & Context Usage
Prioritize Raw Text: If you encounter any discrepancies between the provided \`<metadata>\` and the actual Japanese text, always treat the raw Japanese text as the ultimate source of truth. Ignore any metadata that directly contradicts the text itself.
Context is Crucial: Meticulously utilize the provided \`<metadata>\` (character information, glossary, dictionary) and the preceding lines. This combined context is vital for:
- Maintaining consistency in terminology and characterization.
- Understanding character relationships and the flow of conversation.
- Resolving ambiguities in meaning.
- Inferring the subjects in the sentences, which are often omitted in Japanese by closely examining the preceding few sentences

### Core Translation Directives
Tone and Style Preservation: Faithfully replicate the original author's style and the specific tone of the scene (e.g., humorous, dramatic, romantic, tense).
Dialogue Handling:
- Dialogue lines are enclosed in Japanese quotation marks (e.g., 「 」, 『 』).
- Use the preceding lines and metadata to determine who is speaking the current dialogue line. Assume speakers often alternate in back-and-forth conversation unless context indicates otherwise.
- In the translation, replace them with smart quotation marks (“”).
Pronoun Usage:Ensure correct English pronouns (he, she, it, they, etc.) are used. You should use character information from metadata if they are available.
Narrative Voice: ${instructions.narrative}
Interpret Parentheses: Text within parentheses \`()\` might originate from HTML ruby annotations (furigana) or be authorial asides. Interpret their function contextually. Omit them in the translation, if they are purely phonetic (furigana).
Natural English: Prioritize fluent, natural-sounding English. Avoid overly literal translations. Adapt sentence structure as needed while preserving meaning and intent.

### Names
${instructions.honorifics}
${instructions.nameOrder}

### Output Format
Translation Encapsulation: You MUST place your translated English sentence(s) with \`<translation>\` and \`</translation>\` tags. The extraction script relies strictly on this format.
<example>
- **Input is Dialogue**
Input: \`「ただいま戻りました」\`
Output: <translation>“I have returned.”</translation>
- **Input is Narration/Description**
Input: \`空は青く澄み渡っていた。\`
Output: <translation>The sky was clear and blue.</translation>
</example>

For non Japanese text, simply repeat them back as it is. Your response will be used to replace the original text by a script.
<example>
Input: \`==--==--==\`
Output: <translation> ==--==--== <translation>
</example>
</instructions>
`.trim();

    // --- For Chunking ---
    this.baseChunkingSystem = `
You are an expert text analyst specializing in literary structure. Your primary task is to segment a long-form Japanese text into semantically coherent chunks, preparing it for a downstream translation process. The goal is to create chunks that are logical units of meaning, such as a complete scene, a distinct block of dialogue, or a self-contained descriptive passage.

### Objective
- Given a batch of numbered paragraphs [Start..End], output a single JSON array of [start, end] integer pairs that define contiguous, non-overlapping chunks optimized for translation coherence.

### Input
- <text> contains one paragraph per line, each prefixed by an index like: [123] [content]
  - The [n] prefix is not part of the original source; it is authoritative for paragraph numbering.
  - Paragraphs may be empty; they still count as paragraphs. This are the line breaks in the original text.
  - Content may include Japanese text, ASCII art, site chrome, chat logs, status tables, headings, and formatting artifacts.
- <metadata> contains Start and End indices. These are inclusive bounds for this run and may not begin at 1.

### Output
- A single JSON array: [[a1, b1], [a2, b2], ..., [ak, bk]]
- Constraints:
  - Coverage: The intervals must exactly cover every paragraph index from Start to End with no gaps and no overlaps.
  - Indices must be integers that appear in this input’s [Start..End].
- Output must contain only the JSON array; no commentary, no code fences.

### Chunking Goals
- Primary constraint: chunk length (count content only; exclude the [n] prefixes).
  - Target: 100–200 characters.
  - Allowed: 50–400 characters.
  - Hard cap: Avoid intervals whose content length exceeds 300 characters. Split as needed to respect this cap, even within a long scene or special block.
- Semantic coherence is important. Prefer clean breakpoints, but never exceed the hard cap to preserve a scene.
- Smaller chunks are acceptable; try to avoid ultra‑short chunks (<40 characters) by merging with an adjacent chunk if it remains ≤200 characters.
- Prefer to start chunks at natural breakpoints:
  - Scene or section separators (e.g., ＊＊＊, ─────, =====, ※※※).
  - Headings and metadata lines (e.g., 第N話, 【タイトル】, ◇～視点, side:, 視点：, POV).
  - Clear transitions (time/place/POV/topic), e.g., 翌朝, 数時間後, 一方その頃, やがて.
  - Switches between dialogue-dense blocks and narration blocks, or vice versa.
  - The appearance of special formatted blocks, such as item descriptions, character status screens, system messages in a game-like world, or excerpts from logs/letters.
  - Before and after self-contained “block types” (chat logs, songs, status panels, lists, letters, poems).
- It’s acceptable—and often required—to split long conversations, long narration, chat logs, lists, or tables across multiple chunks to satisfy the character budget. Choose the least disruptive boundary (after sentence-ending punctuation 「。！？」、after closing quotes 」/』、at paragraph breaks、or between list/log/table rows).

### Do not split inside the following unless unavoidable
- Prefer to keep these contiguous, but if keeping them intact would cause a chunk to exceed 200 characters, you must split within the block. Use these safe sub-boundaries:
  - Continuous dialogue: between utterances (between lines starting with 「 or 『) or after a narration beat; avoid cutting inside a single speech line if possible.
  - Lists/enumerations: between items.
  - Chat/comment logs: between messages; group a handful of lines per chunk to meet the budget.
  - Status/character sheets or tables: between rows or labeled subsections.
  - Poems/songs/incantations and ASCII-art: between stanzas/lines or visually separable segments; avoid breaking a single line.
  - Letters/emails/notes: at paragraph breaks; avoid mid-sentence if there’s any alternative.
  
### Edges and overlaps
- The first and last paragraphs in this window may be truncated mid-sentence due to batching. Still ensure coverage [Start..End]; prefer placing boundaries exactly at Start and End rather than guessing beyond the visible window.
- Do not invent or renumber indices. Do not reference paragraphs outside [Start..End].

### Examples

- Example 1: Scene break
Input Snippet:
\`\`\`
...
[42] 【文】...【文】
[43]
[44] 【文】...【文】
[45]
[46] 【文】
[47] ◆◆◆
[48] 【文】...【文】
[49]
...
\`\`\`
Potential Output Snippet: \`..., [39, 46], [47, 56], ...\`

- Example 2: Dialogue to Narration Shift
Input Snippet:
\`\`\`
...
[21] 「台詞」
[22] 「台詞」
[23] 「台詞」
[24] 【文】...【文】
[25] 【文】
...
\`\`\`
Potential Output Snippet: \`..., [15, 23], [24, 32], ...\`

- Example 3: Special Content Block
Input Snippet:
\`\`\`
...
[77] 【文】...【文】
[78] ▼ステータス
[79] 名前：【名前】
[80] レベル：5
...
[84] INT: 120
[85] ▲
[86] 【文】...
...
\`\`\`
Potential Output Snippet: \`..., [65, 77], [78, 85], [86, 95], ...\`
`.trim();
  }

  /**
   * Generates the complete prompt for a translation task.
   * @param {string} textToTranslate - The main text block to be translated.
   * @param {string} precedingText - The context from previous lines.
   * @returns {{system: string, user: string}}
   */
  getTranslationPrompt(textToTranslate, precedingText = '') {
    const fullContextText = precedingText + textToTranslate;
    const dictionaryMetadata = this.dictionary ? generateMetadata(fullContextText, this.dictionary) : '';
    const precedingTextContext = precedingText ? `\nHere are the lines immediately preceding the text to be translated, for context:\n${precedingText}` : '';

    const metadataBlock = (dictionaryMetadata || precedingTextContext)
                          ? `<metadata>\n${dictionaryMetadata}\n</metadata>\n${precedingTextContext}`
                          : '';

    const customInstructions = (this.uiConfig.customInstruction)
                               ? `
### Additional Notes:
${this.uiConfig.customInstruction}
      `.trim()
                               : '';

    const user = [
      customInstructions,
      metadataBlock,
      `Translate the following Japanese text into English:\n${textToTranslate}`,
    ].filter(Boolean).join('\n\n');

    return {
      system: this.baseTranslationSystem + '\n\n' + this.baseTranslationDev,
      user: user,
    };
  }

  /**
   * Generates the prompt for a chunking task.
   * @param {Array<{text: string, index: number}>} indexedParagraphs - The paragraphs for this batch.
   * @param offset - The offset to subtract for mapping indices to lower range.
   * @returns {{system: string, user: string}}
   */
  getChunkingPrompt(indexedParagraphs, offset = 0) {
    const textBlock = indexedParagraphs
      .map(p => `[${p.index + 1 - offset}] ${p.text}`)
      .join('\n');
    const startIndex = indexedParagraphs[0]?.index + 1 - offset || 1;
    const endIndex = indexedParagraphs[indexedParagraphs.length - 1]?.index + 1 - offset || 1;

    const userPrompt = `
<text>
${textBlock}
</text>
<metadata>
Start: ${startIndex}
End: ${endIndex}
</metadata>
`.trim();

    return {
      system: this.baseChunkingSystem,
      user: userPrompt,
    };
  }
}

/**
 * Base class for all translation strategies.
 * Defines the common interface for executing a translation process.
 */
class TranslationStrategy {
  /**
   * @param {object} uiConfig - The configuration object from the UI.
   * @param {PromptManager} promptManager - The stateful prompt builder for the session.
   */
  constructor(uiConfig, promptManager) {
    this._chunkQueue = null;
    this._translationQueue = null;
    this.uiConfig = uiConfig;
    this.promptManager = promptManager;
  }

  get chunkQueue() {
    if (!this._chunkQueue) {
      this._chunkQueue = createStageQueue(this.uiConfig.stage3a, 'Text Chunking');
    }
    return this._chunkQueue;
  }

  get translationQueue() {
    if (!this._translationQueue) {
      this._translationQueue = createStageQueue(this.uiConfig.stage3b, 'Translation');
    }
    return this._translationQueue;
  }

  /**
   * Executes the translation strategy.
   * This method must be implemented by concrete strategy classes.
   * @param {Map<string, string>} paragraphMap - A map of element ID to text content.
   * @returns {Promise<void>} A promise that resolves when the process is complete.
   */
  async execute(paragraphMap) {
    throw new Error('Strategy must implement the execute() method.');
  }

  processResponse(rawLLMResponse) {
    return extractTextFromTag(rawLLMResponse, 'translation')
      .replace(/\n+/g, '\n') // Compresses newlines chars (2+ -> 1)
      .split('\n')
      .map(textPostProcess)
      .join('\n');
  }
}

/**
 * A strategy that translates each paragraph (<p> element) individually,
 * sending one API request per line.
 */
class SingleLineStrategy extends TranslationStrategy {
  /**
   * @param {object} uiConfig - The configuration object from the UI.
   * @param {PromptManager} promptManager - The stateful prompt builder for the session.
   */
  constructor(uiConfig, promptManager) {
    super(uiConfig, promptManager);
  }

  async execute(paragraphMap) {
    const numPreviousLines = this.uiConfig.contextLines;
    const tasks = [];
    const paragraphs = Array.from(paragraphMap.entries())
      .filter(entry => entry[1]);

    for (let i = 0; i < paragraphs.length; i++) {
      const [id, text] = paragraphs[i];

      // Gather the preceding lines for context
      const precedingLines = paragraphs
        .slice(Math.max(0, i - numPreviousLines), i)
        .map(p => p[1])
        .join('\n');

      const prompt = this.promptManager.getTranslationPrompt(text, precedingLines);

      const task = this.translationQueue.enqueueTask(prompt, (result) => {
        if (result.ok) {
          const translatedText = this.processResponse(result.response);
          updateParagraphContent(id, translatedText);
        } else {
          console.error(`Failed to translate paragraph ${id}:`, result.error);
        }
      });
      tasks.push(task);
    }

    await Promise.all(tasks);
    this.translationQueue.dispose();
  }
}

class ChunkingStrategy extends TranslationStrategy {
  /**
   * @param {object} uiConfig - The configuration object from the UI.
   * @param {PromptManager} promptManager - The stateful prompt builder for the session.
   */
  constructor(uiConfig, promptManager) {
    super(uiConfig, promptManager);
  }

  async execute(paragraphMap) {
    const numPreviousLines = this.uiConfig.contextLines;

    const allParagraphs = Array.from(paragraphMap.entries())
      .map(([id, text], index) => ({ id, text, index }));

    // ----------------------------------------
    //  Part 1: Generate chunking suggestions in overlapping batches

    const chunkingBatches = [];
    let currentStartIndex = 0;

    while (currentStartIndex < allParagraphs.length) {
      let currentChars = 0;
      let endIndex = currentStartIndex;

      // Greedily build a batch until we reach the character limit
      for (let i = currentStartIndex; i < allParagraphs.length; i++) {
        const p = allParagraphs[i];
        if (p.text.length > 1000) {
          console.warn(`Paragraph ${p.index + 1} (id: ${p.id}) is very long (${p.text.length} chars). This might affect chunking quality.`);
        }
        if (currentChars > 0 && currentChars + p.text.length > BATCH_CHAR_LIMIT) {
          break;
        }
        currentChars += p.text.length;
        endIndex = i;
      }

      const batch = allParagraphs.slice(currentStartIndex, endIndex + 1);
      const actualStart = batch[0].index + 1;
      const actualEnd = batch[batch.length - 1].index + 1;

      // Calculate offset: map to start at 20 if actualStart >= 20, otherwise no offset
      const offset = actualStart >= 20 ? actualStart - 20 : 0;

      chunkingBatches.push({
        batch,
        offset,
        actualStart,
        actualEnd,
      });

      // If we've processed the last paragraph, our work is done here
      if (endIndex >= allParagraphs.length - 1) {
        break;
      }

      // Calculate the start of the next batch, ensuring we always move forward
      const nextStartIndex = endIndex - OVERLAP_PARAGRAPH_COUNT + 1;
      currentStartIndex = Math.max(currentStartIndex + 1, nextStartIndex);
    }

    const chunkingPromises = chunkingBatches.map(({ batch, offset }) => {
      const prompt = this.promptManager.getChunkingPrompt(batch, offset);
      return this.chunkQueue.enqueueTask(prompt);
    });

    const chunkingResults = await Promise.all(chunkingPromises);
    this.chunkQueue.dispose();

    // ----------------------------------------
    //  Part 2: Merge the fuzzy suggestions into clean intervals

    const llmSuggestions = [];
    for (let i = 0; i < chunkingResults.length; i++) {
      const result = chunkingResults[i];
      const { offset, actualStart, actualEnd } = chunkingBatches[i];

      if (result.ok) {
        try {
          const suggestions = JSON.parse(result.response);
          if (Array.isArray(suggestions) && suggestions.length > 0) {
            // Validate and reverse the mapping
            const unmappedSuggestions = [];
            let isValid = true;

            for (const interval of suggestions) {
              // Validate interval structure
              if (!Array.isArray(interval) || interval.length !== 2) {
                isValid = false;
                break;
              }

              const [start, end] = interval;

              if (typeof start !== 'number' || typeof end !== 'number') {
                isValid = false;
                break;
              }

              // Reverse the mapping
              const unmappedStart = start + offset;
              const unmappedEnd = end + offset;

              // Validate that unmapped values are within expected range
              if (unmappedStart < actualStart || unmappedEnd > actualEnd || unmappedStart > unmappedEnd) {
                console.warn(`Interval [${unmappedStart}, ${unmappedEnd}] out of range [${actualStart}, ${actualEnd}] for batch ${i}`);
                isValid = false;
                break;
              }

              unmappedSuggestions.push([unmappedStart, unmappedEnd]);
            }

            if (isValid) {
              llmSuggestions.push(unmappedSuggestions);
            } else {
              console.warn(`Invalid interval structure for batch ${i}, pushing empty array`);
              llmSuggestions.push([]);
            }
          } else {
            // Empty or invalid suggestions array
            llmSuggestions.push([]);
          }
        } catch (e) {
          console.warn(`Invalid LLM response format for batch ${i}:`, result.response, e);
          llmSuggestions.push([]);
        }
      } else {
        console.error(`Chunking request for batch ${i} failed:`, result.error);
        llmSuggestions.push([]);
      }
    }

    const finalIntervals = mergeFuzzyIntervals({
      totalParagraphs: allParagraphs.length,
      llmSuggestions: llmSuggestions,
    });

    // ----------------------------------------
    //  Part 3: Translate using the final, merged intervals

    const translationTasks = [];
    for (const interval of finalIntervals) {
      const [start, end] = [interval[0] - 1, interval[1]]; // Intervals are 1-indexed
      const chunkParagraphs = allParagraphs.slice(start, end);

      if (chunkParagraphs.length === 0) continue;

      // Build context from preceding lines
      const precedingLines = allParagraphs
        .slice(Math.max(0, start - numPreviousLines), start)
        .map(p => p.text)
        .join('\n');

      // Join chunk paragraphs with newlines for translation
      const chunkText = chunkParagraphs.map(p => p.text).join('\n');

      const prompt = this.promptManager.getTranslationPrompt(chunkText, precedingLines);

      const task = this.translationQueue.enqueueTask(prompt, (result) => {
        if (result.ok) {
          const translatedText = this.processResponse(result.response);
          const translatedLines = translatedText.split('\n');

          const numExpected = chunkParagraphs.length;
          const numReceived = translatedLines.length;

          if (numReceived === numExpected) {
            // Ideal case: one-to-one mapping
            for (let i = 0; i < numExpected; i++) {
              updateParagraphContent(chunkParagraphs[i].id, translatedLines[i]);
            }

          } else if (numReceived > numExpected) {
            // More lines than paragraphs: map first n-1, join extras into last
            for (let i = 0; i < numExpected - 1; i++) {
              updateParagraphContent(chunkParagraphs[i].id, translatedLines[i]);
            }
            const extraLines = translatedLines.slice(numExpected - 1);
            const wrappedLines = extraLines.map(line => `<span>${line}</span>`);
            const combinedContent = wrappedLines.join('<br>');
            updateParagraphContent(chunkParagraphs[numExpected - 1].id, combinedContent);

          } else {
            // Fewer lines than paragraphs: map available lines, hide the rest
            for (let i = 0; i < numReceived; i++) {
              updateParagraphContent(chunkParagraphs[i].id, translatedLines[i]);
            }
            for (let i = numReceived; i < numExpected; i++) {
              const elem = document.getElementById(chunkParagraphs[i].id);
              if (elem) {
                elem.style.display = 'none';
              }
            }
          }
        } else {
          console.error(`Failed to translate chunk:`, result.error);
        }
      });

      translationTasks.push(task);
    }

    await Promise.all(translationTasks);
    this.translationQueue.dispose();
  }
}

/**
 * Combines all text and sends a single request.
 */
class EntirePageStrategy extends TranslationStrategy {
  /**
   * @param {object} uiConfig - The configuration object from the UI.
   * @param {PromptManager} promptManager - The stateful prompt builder for the session.
   */
  constructor(uiConfig, promptManager) {
    super(uiConfig, promptManager);
  }

  async execute(paragraphMap) {
    // Sort by numeric order. id should end in a number [...01, ...02, ...03] or [...1, ...2, ...10, ...11]
    const combinedRawText = Array.from(paragraphMap.entries())
      .sort((a, b) => a[0].localeCompare(b[0], undefined, { numeric: true }))
      .map(([_, v]) => v)
      .join('\n');

    const prompt = this.promptManager.getTranslationPrompt(combinedRawText, '');

    const task = this.translationQueue.enqueueTask(prompt, (result) => {
      if (result.ok) {
        const translatedText = this.processResponse(result.response);
        const formattedText = translatedText.replace(/\r\n|\r|\n/g, '<br>');


        const [firstKey, ...remaining] = paragraphMap.keys();
        updateParagraphContent(firstKey, formattedText);
        remaining.forEach(id => {
          const elem = document.getElementById(id);
          if (elem) {
            elem.style.display = 'none';
          }
        });
      } else {
        console.error(`Failed to translate paragraph ${id}:`, result.error);
      }
    });

    tasks.push(task);
    await Promise.all(tasks);
    this.translationQueue.dispose();
  }
}

/* ================================================================================
   Utility functions for translation stage
   ================================================================================*/

/**
 * Generates metadata text based on the provided dictionary and input text.
 * @param {string} text - The input text to analyze.
 * @param {object} dictionary - The glossary dictionary with entries.
 * @returns {string} The formatted metadata string or an empty string if no matches.
 */
function generateMetadata(text, dictionary) {
  const metadataArray = [];

  for (const entry of dictionary.entries) {
    if (entry.keys.some(key => text.includes(key))) {
      metadataArray.push(entry.value);
    }
  }

  if (metadataArray.length !== 0) {
    return metadataArray.join('\n');
  } else {
    return '';
  }
}

/**
 * Merge fuzzy LLM-suggested paragraph intervals into clean, contiguous chunks.
 * Steps:
 *  1) Normalize suggestions, dedupe, clamp.
 *  2) Boundary voting to fuse fuzzy cut points.
 *  3) Build intervals and sanitize for gaps/overlaps with warnings.
 *  4) Robust fallback to equal-sized chunks if LLM output unusable.
 *
 * Options:
 *  - totalParagraphs (required): N
 *  - llmSuggestions (required): nested arrays of [start, end] pairs from the LLM (any nesting ok)
 *  - fuzz (default 2): radius in paragraphs to cluster “nearby” boundaries
 *  - fallbackSize (default 60): paragraph size for last-resort fixed chunking
 *
 * Returns: Array<[start, end]> covering [1..N] with no gaps/overlaps.
 */
const mergeFuzzyIntervals = ({ totalParagraphs, llmSuggestions, fuzz = 2, fallbackSize = 60 }) => {
  console.log(`Merging the following intervals: ${JSON.stringify(llmSuggestions)}`);
  const N = Number(totalParagraphs || 0);
  if (!Number.isInteger(N) || N <= 0) {
    throw new Error("totalParagraphs must be a positive integer");
  }

  // ---------------------
  // Fallback chunker
  // Used if we cannot parse/normalize any LLM suggestions.
  const chunkEveryK = (size) => {
    const k = Math.max(1, Number(size || 1));
    const out = [];
    for (let start = 1; start <= N; start += k) {
      const end = Math.min(N, start + k - 1);
      out.push([start, end]);
    }
    return out;
  };

  // ---------------------
  // Utilities

  const isNumeric = (v) => typeof v === "number" && Number.isFinite(v);
  const toInt = (v) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.trunc(n) : NaN;
  };

  // Recursively collect [s,e] pairs from any nested array structure.
  const collectPairs = (node, acc) => {
    if (!Array.isArray(node)) return;
    if (node.length === 2 && (isNumeric(node[0]) || isNumeric(toInt(node[0]))) && (isNumeric(node[1]) || isNumeric(toInt(node[1])))) {
      const s = toInt(node[0]);
      const e = toInt(node[1]);
      if (Number.isInteger(s) && Number.isInteger(e)) acc.push([s, e]);
      return;
    }
    for (const child of node) collectPairs(child, acc);
  };

  // Normalize: clamp, swap if reversed, drop out-of-range, dedupe.
  const normalizeIntervals = (pairs) => {
    const norm = [];
    for (let [s, e] of pairs) {
      if (!Number.isInteger(s) || !Number.isInteger(e)) continue;
      if (s > e) [s, e] = [e, s];
      // Clamp to [1, N]
      s = Math.max(1, Math.min(N, s));
      e = Math.max(1, Math.min(N, e));
      if (s > e) continue;
      norm.push([s, e]);
    }
    // Deduplicate
    const key = ([a, b]) => `${a},${b}`;
    const seen = new Set();
    const dedup = [];
    for (const p of norm) {
      const k = key(p);
      if (!seen.has(k)) {
        dedup.push(p);
        seen.add(k);
      }
    }
    // Sort by start, then end
    dedup.sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    return dedup;
  };

  // Build boundary vote counts from intervals:
  // Boundary k means a cut after paragraph k (k in [0..N]).
  const buildBoundaryVotes = (intervals) => {
    const counts = new Uint32Array(N + 1);
    for (const [s, e] of intervals) {
      counts[s - 1] += 1; // cut before s
      counts[e] += 1;     // cut after e
    }
    return counts;
  };

  // Cluster nearby boundaries within "fuzz" and pick one representative per cluster.
  // Representative is chosen by weighted mean (vote strength), snapped to the nearest
  // member of the cluster; ties broken by higher vote, then lower index.
  const selectBoundariesFromVotes = (counts, fuzzRadius) => {
    const positions = [];
    for (let k = 0; k <= N; k++) {
      if (counts[k] > 0) positions.push(k);
    }

    if (positions.length === 0) {
      // No proposed cuts; return only [0, N]
      return [0, N];
    }

    const clusters = [];
    let cur = [positions[0]];
    for (let i = 1; i < positions.length; i++) {
      const p = positions[i];
      const last = cur[cur.length - 1];
      if (p - last <= fuzzRadius) {
        cur.push(p);
      } else {
        clusters.push(cur);
        cur = [p];
      }
    }
    clusters.push(cur);

    const chosen = [];
    for (const cluster of clusters) {
      if (cluster.length === 1) {
        chosen.push(cluster[0]);
        continue;
      }
      let sumW = 0;
      let sumWP = 0;
      for (const p of cluster) {
        const w = counts[p] || 0;
        sumW += w;
        sumWP += w * p;
      }
      const mean = sumW > 0 ? Math.round(sumWP / sumW) : Math.round(cluster.reduce((a, b) => a + b, 0) / cluster.length);

      // Snap to nearest cluster member; tie-break by higher count, then lower index
      let best = cluster[0];
      let bestScore = Number.POSITIVE_INFINITY;
      let bestVotes = counts[best] || 0;
      for (const p of cluster) {
        const dist = Math.abs(p - mean);
        const v = counts[p] || 0;
        if (
          dist < bestScore ||
          (dist === bestScore && v > bestVotes) ||
          (dist === bestScore && v === bestVotes && p < best)
        ) {
          best = p;
          bestScore = dist;
          bestVotes = v;
        }
      }
      chosen.push(best);
    }

    // Ensure 0 and N exist
    if (!chosen.includes(0)) chosen.push(0);
    if (!chosen.includes(N)) chosen.push(N);

    // Sort and unique
    chosen.sort((a, b) => a - b);
    const uniq = [];
    for (const k of chosen) {
      if (uniq.length === 0 || uniq[uniq.length - 1] !== k) uniq.push(k);
    }
    return uniq;
  };

  const intervalsFromBoundaries = (boundaries) => {
    const out = [];
    for (let i = 0; i < boundaries.length - 1; i++) {
      const s = boundaries[i] + 1;
      const e = boundaries[i + 1];
      if (s <= e) out.push([s, e]);
    }
    return out;
  };

  // Fix gaps/overlaps to produce a strict partition of [1..N].
  // - Overlap: delete the overlap from the latter interval (shift its start).
  // - Gap: extend the previous interval to fill.
  // Logs console warnings when adjustments occur.
  const sanitizeContiguity = (intervals) => {
    // Sort defensively
    const sorted = [...intervals].sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
    const fixed = [];
    let expectedStart = 1;

    for (let idx = 0; idx < sorted.length; idx++) {
      let [s, e] = sorted[idx];

      if (e < expectedStart) {
        // This interval is entirely covered; drop it silently.
        continue;
      }

      if (s > expectedStart) {
        // Gap before this interval; extend previous if possible, or synthesize if none
        if (fixed.length > 0) {
          const prev = fixed[fixed.length - 1];
          console.warn(`[chunk-merge] Gap detected: extending previous ${prev} to ${prev[0]},${s - 1}`);
          prev[1] = s - 1;
        } else {
          // Leading gap
          console.warn(`[chunk-merge] Leading gap detected: synthesizing [${expectedStart}, ${s - 1}]`);
          fixed.push([expectedStart, s - 1]);
        }
      }

      if (s < expectedStart) {
        // Overlap: trim the latter interval
        console.warn(`[chunk-merge] Overlap detected: trimming start of [${s}, ${e}] to ${expectedStart}`);
        s = expectedStart;
      }

      // Clamp after adjustments
      s = Math.max(1, Math.min(N, s));
      e = Math.max(1, Math.min(N, e));
      if (s <= e) {
        fixed.push([s, e]);
        expectedStart = e + 1;
      }
    }

    // Tail coverage if needed
    if (expectedStart <= N) {
      console.warn(`[chunk-merge] Trailing gap detected: synthesizing [${expectedStart}, ${N}]`);
      fixed.push([expectedStart, N]);
    }

    // Final coalescing for any accidental adjacency artifacts
    const coalesced = [];
    for (const [s, e] of fixed) {
      if (coalesced.length === 0) {
        coalesced.push([s, e]);
        continue;
      }
      const last = coalesced[coalesced.length - 1];
      if (last[1] + 1 === s) {
        // Keep as separate chunks; adjacency is OK and intentional
        coalesced.push([s, e]);
      } else if (last[1] >= s) {
        // Safety: ensure no residual overlap
        last[1] = Math.max(last[1], e);
      } else {
        coalesced.push([s, e]);
      }
    }

    return coalesced;
  };

  // ---------------------
  // Pipeline

  // 1) Parse & normalize LLM suggestions
  const rawPairs = [];
  collectPairs(llmSuggestions, rawPairs);

  if (rawPairs.length === 0) {
    // Hard fallback
    console.warn('Unable to read LLM splitting suggestions, using fallback.');
    return chunkEveryK(fallbackSize);
  }

  const intervals = normalizeIntervals(rawPairs);
  if (intervals.length === 0) {
    console.warn('Unable to read LLM splitting suggestions, using fallback.');
    return chunkEveryK(fallbackSize);
  }

  // 2) Boundary voting and fuzzy consolidation
  const counts = buildBoundaryVotes(intervals);
  const boundaries = selectBoundariesFromVotes(counts, fuzz);

  // 3) Build intervals and sanitize contiguity
  const prelim = intervalsFromBoundaries(boundaries);
  const finalChunks = sanitizeContiguity(prelim);

  // If everything went sideways (extremely unlikely), fall back.
  if (finalChunks.length === 0) {
    console.warn('Unable to read LLM splitting suggestions, using fallback.');
    return chunkEveryK(fallbackSize);
  }

  console.log(`Merged intervals for splitting: ${JSON.stringify(finalChunks)}`);
  return finalChunks;
};

/* ================================================================================
   Main process pipeline
   ================================================================================*/

async function main() {
  // Retrive settings and config values from the UI
  const uiConfig = UIConfigValidator.validate({ modelsList });

  if (uiConfig === null) {
    console.error('[Translation] Configuration validation failed. Cannot proceed.');
    return;
  }

  // Site specific logic
  const domainManager = new DomainManager();
  const texts = extractTextFromWebpage(domainManager);

  // Stage 1 & 2: Generate new glossary entries with raw text
  if (uiConfig.stage1 && uiConfig.stage2) {
    const glossaryManager = new GlossaryManager(uiConfig);
    const updatedDictionary = await glossaryManager.generateAndUpdateDictionary(loadDictionary(domainManager), texts);

    // Save updated dictionary to storage
    saveDictionary(domainManager, updatedDictionary);
  }

  const dictionary = loadDictionary(domainManager);

  // Start translation process
  await translateTexts(dictionary, uiConfig, texts);

  // Done
}

async function translateTexts(dictionary, uiConfig, paragraphMap) {
  const promptManager = new PromptManager(uiConfig, dictionary);

  let strategy;

  switch (uiConfig.translationMethod) {
    case 'chunk':
      strategy = new ChunkingStrategy(uiConfig, promptManager);
      break;
    case 'single':
      strategy = new SingleLineStrategy(uiConfig, promptManager);
      break;
    case 'entire':
      strategy = new EntirePageStrategy(uiConfig, promptManager);
      break;
    default:
      throw new Error(`Unknown translation method: ${uiConfig.translationMethod}`);
  }

  await strategy.execute(paragraphMap);
}

/* ================================================================================
   UI
   ================================================================================*/

class UIConfigValidator {
  static schema = {
    contextLines: {
      type: 'number',
      min: 1,
      max: 10,
      default: 3,
      selector: '#context-lines-input',
    },
    narrative: {
      type: 'enum',
      allowed: ['first', 'third', 'auto'],
      default: 'auto',
      selector: '#narrative-select',
    },
    honorifics: {
      type: 'enum-nullable',
      allowed: ['preserve', 'nil'],
      default: 'nil',
      selector: '#honorifics-checkbox',
      transform: (checked) => checked ? 'preserve' : 'nil',
    },
    nameOrder: {
      type: 'enum',
      allowed: ['en', 'jp'],
      default: 'jp',
      selector: '#name-order-select',
    },
    customInstruction: {
      type: 'string-nullable',
      default: null,
      selector: '#custom-instruction-textarea',
      transform: (value) => value.trim() || null,
    },
    translationMethod: {
      type: 'enum',
      allowed: ['chunk', 'single', 'entire'],
      default: 'chunk',
      selector: '#translation-method-select',
    },
    stage1: {
      type: 'enum-nullable',
      allowedFrom: 'modelsList',
      default: null,
      selector: '#stage1-select',
      transform: (value) => value || null,
    },
    stage2: {
      type: 'enum-nullable',
      allowedFrom: 'modelsList',
      default: null,
      selector: '#stage2-select',
      transform: (value) => value || null,
    },
    stage3a: {
      type: 'enum-nullable',
      allowedFrom: 'modelsList',
      default: null,
      selector: '#stage3a-select',
      transform: (value) => value || null,
    },
    stage3b: {
      type: 'enum-nullable',
      allowedFrom: 'modelsList',
      default: null,
      selector: '#stage3b-select',
      transform: (value) => value || null,
    },
  };

  // Cross-field validation rules
  static crossFieldRules = [
    {
      name: 'stage1-stage2-dependency',
      validate: (config) => {
        const stage1Set = config.stage1 !== null;
        const stage2Set = config.stage2 !== null;

        if (stage1Set !== stage2Set) {
          return {
            isValid: false,
            message: 'stage1 and stage2 must both be set, or both be unset',
          };
        }
        return { isValid: true };
      },
    },
    {
      name: 'stage3a-chunk-requirement',
      validate: (config) => {
        if (config.translationMethod === 'chunk' && config.stage3a === null) {
          return {
            isValid: false,
            message: 'stage3a must be set when translationMethod is "chunk"',
          };
        }
        return { isValid: true };
      },
    },
    {
      name: 'stage-model-limits-compliance',
      validate: (config) => {
        const stages = ['stage1', 'stage2', 'stage3a', 'stage3b'];
        const availableModels = getAvailableModels();

        for (const stage of stages) {
          const modelId = config[stage];

          // Skip if no model selected for this stage
          if (modelId === null) continue;

          // Find the model in available models
          const selectedModel = availableModels.find(m => m.id === modelId);
          if (!selectedModel) {
            return {
              isValid: false,
              message: `Selected model '${modelId}' for '${stage}' is not available`,
            };
          }

          // Check if model is allowed for this stage
          const providerConfig = PROVIDER_CONFIGS[selectedModel.provider];
          const stageLimits = providerConfig.limits[stage];

          if (stageLimits === undefined) {
            console.warn(`[UIConfigValidator] Stage '${stage}' not found in limits config for provider '${selectedModel.provider}'`);
            continue;
          }

          if (stageLimits !== 'all' && (!Array.isArray(stageLimits) || !stageLimits.includes(modelId))) {
            return {
              isValid: false,
              message: `Model '${selectedModel.label}' (${modelId}) is not allowed for '${stage}'`,
            };
          }
        }

        return { isValid: true };
      },
    },
  ];

  /**
   * Validates the UI configuration.
   * @param {object} dynamicLists - An object containing dynamically generated lists, e.g., { modelsList: [...] }
   * @returns {object|null} The validated configuration object, or null if validation fails critically.
   */
  static validate(dynamicLists = {}) {
    const config = {};
    const warnings = [];

    for (const [key, spec] of Object.entries(this.schema)) {
      try {
        const element = document.querySelector(spec.selector);
        let value;

        if (!element) {
          warnings.push(`UI element for '${key}' not found (${spec.selector}), using default: ${spec.default}`);
          value = spec.default;
        } else {
          value = this.extractValue(element, spec);
        }

        if (spec.transform) {
          value = spec.transform(value);
        }

        // Determine the allowed values: from dynamicLists, from static spec.allowed, or undefined
        const allowedValues = spec.allowedFrom
                              ? dynamicLists[spec.allowedFrom]
                              : spec.allowed;

        const validatedValue = this.validateValue(key, value, spec, allowedValues);

        if (validatedValue.isValid) {
          config[key] = validatedValue.value;
        } else {
          warnings.push(`Invalid value for '${key}': ${value}. ${validatedValue.reason}. Using default: ${spec.default}`);
          config[key] = spec.default;
        }
      } catch (error) {
        warnings.push(`Error processing '${key}': ${error.message}. Using default: ${spec.default}`);
        config[key] = spec.default;
      }
    }

    if (warnings.length > 0) {
      console.warn('[UIConfig] Validation warnings:', warnings);
    }

    for (const rule of this.crossFieldRules) {
      const result = rule.validate(config);
      if (!result.isValid) {
        console.error(`[UIConfig] Cross-field validation failed (${rule.name}): ${result.message}`);
        return null;
      }
    }

    return config;
  }

  static extractValue(element, spec) {
    if (element.type === 'checkbox') {
      return element.checked;
    } else if (element.tagName === 'SELECT') {
      return element.value;
    } else if (element.tagName === 'INPUT') {
      return spec.type === 'number' ? parseFloat(element.value) : element.value;
    } else if (element.tagName === 'TEXTAREA') {
      return element.value;
    }
    return element.value;
  }

  static validateValue(key, value, spec, allowedValues) {
    switch (spec.type) {
      case 'number':
        if (typeof value !== 'number' || isNaN(value)) {
          return { isValid: false, reason: 'Not a valid number' };
        }
        if (spec.min !== undefined && value < spec.min) {
          return { isValid: false, reason: `Must be >= ${spec.min}` };
        }
        if (spec.max !== undefined && value > spec.max) {
          return { isValid: false, reason: `Must be <= ${spec.max}` };
        }
        return { isValid: true, value };

      case 'enum':
        if (!allowedValues.includes(value)) {
          return {
            isValid: false,
            reason: `Must be one of: ${allowedValues.join(', ')}`,
          };
        }
        return { isValid: true, value };

      case 'enum-nullable':
        if (value !== null && !allowedValues.includes(value)) {
          return {
            isValid: false,
            reason: `Must be one of: ${allowedValues.join(', ')} or null`,
          };
        }
        return { isValid: true, value };

      case 'string-nullable':
        if (value !== null && typeof value !== 'string') {
          return { isValid: false, reason: 'Must be a string or null' };
        }
        return { isValid: true, value };

      default:
        return { isValid: true, value };
    }
  }
}

function getAvailableModels() {
  const availableModels = [];

  for (const [provider, config] of Object.entries(PROVIDER_CONFIGS)) {
    if (PROVIDER_API_CONFIG[provider]?.apiKey !== null) {
      for (const model of config.models) {
        availableModels.push({
          id: model.id,
          provider: provider,
          label: model.label,
          ...model,
        });
      }
    }
  }

  return availableModels;
}

function filterModelsByStage(availableModels, stage) {
  const stageFilteredModels = [];

  for (const model of availableModels) {
    const providerConfig = PROVIDER_CONFIGS[model.provider];
    const stageLimits = providerConfig.limits[stage];

    // If stage doesn't exist in the config, warn and allow all models
    if (stageLimits === undefined) {
      console.warn(`[ModelSelector] Stage '${stage}' not found in limits config for provider '${model.provider}'. Allowing all models.`);
      stageFilteredModels.push(model);
      continue;
    }

    // If 'all' is specified, include the model
    if (stageLimits === 'all') {
      stageFilteredModels.push(model);
    }
    // If it's an array, check if model.id is included
    else if (Array.isArray(stageLimits) && stageLimits.includes(model.id)) {
      stageFilteredModels.push(model);
    }
  }

  return stageFilteredModels;
}

function createModelSelector(elementId, availableModels, stage) {
  const select = document.createElement('select');
  select.id = elementId;
  select.style.cssText = `
    width: 100%;
    padding: 4px;
    margin-top: 4px;
    border: 1px solid #ccc;
    border-radius: 3px;
    background: white;
    font-size: 12px;
  `;

  const defaultOption = document.createElement('option');
  defaultOption.value = '';
  defaultOption.textContent = 'Select...';
  select.appendChild(defaultOption);

  // Filter models based on stage limits if stage is provided
  let filteredModels = availableModels;

  if (stage && typeof stage === 'string') {
    filteredModels = filterModelsByStage(availableModels, stage);
  }

  // Group models by provider
  const modelsByProvider = {};
  for (const model of filteredModels) {
    if (!modelsByProvider[model.provider]) {
      modelsByProvider[model.provider] = [];
    }
    modelsByProvider[model.provider].push(model);
  }

  for (const [provider, models] of Object.entries(modelsByProvider)) {
    const optgroup = document.createElement('optgroup');
    optgroup.label = provider.charAt(0).toUpperCase() + provider.slice(1);

    for (const model of models) {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.label;
      option.dataset.provider = provider;
      optgroup.appendChild(option);
    }

    select.appendChild(optgroup);
  }

  return select;
}

function createCustomInstructionsDialog() {
  const overlay = document.createElement('div');
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: none;
    z-index: 10001;
    align-items: center;
    justify-content: center;
  `;

  const dialog = document.createElement('div');
  dialog.style.cssText = `
    background: white;
    padding: 20px;
    border-radius: 8px;
    width: 500px;
    max-width: 90%;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  `;

  const title = document.createElement('h3');
  title.textContent = 'Custom Translation Instructions';
  title.style.cssText = `
    margin: 0 0 12px 0;
    font-size: 16px;
  `;

  const textarea = document.createElement('textarea');
  textarea.id = 'custom-instruction-textarea';
  textarea.placeholder = 'Enter any additional instructions for the translation...';
  textarea.style.cssText = `
    width: 100%;
    height: 200px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-family: inherit;
    font-size: 13px;
    resize: vertical;
    box-sizing: border-box;
  `;

  const buttonContainer = document.createElement('div');
  buttonContainer.style.cssText = `
    margin-top: 12px;
    text-align: right;
  `;

  const closeButton = document.createElement('button');
  closeButton.textContent = 'Close';
  closeButton.style.cssText = `
    padding: 6px 16px;
    border: none;
    border-radius: 4px;
    background: #007bff;
    color: white;
    cursor: pointer;
    font-size: 13px;
  `;
  closeButton.addEventListener('mouseenter', () => {
    closeButton.style.background = '#0056b3';
  });
  closeButton.addEventListener('mouseleave', () => {
    closeButton.style.background = '#007bff';
  });
  closeButton.addEventListener('click', () => {
    overlay.style.display = 'none';
  });

  buttonContainer.appendChild(closeButton);
  dialog.appendChild(title);
  dialog.appendChild(textarea);
  dialog.appendChild(buttonContainer);
  overlay.appendChild(dialog);

  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      overlay.style.display = 'none';
    }
  });

  return overlay;
}

// Allows expand / collpase toggle for 'Models' and 'Options'
function createCollapsibleSection(title, content) {
  const section = document.createElement('div');
  section.style.cssText = `
    margin-bottom: 8px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
  `;

  const header = document.createElement('div');
  header.style.cssText = `
    padding: 4px;
    background: #f0f0f0;
    cursor: pointer;
    user-select: none;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-radius: 4px;
  `;
  header.addEventListener('mouseenter', () => {
    header.style.background = '#e8e8e8';
  });
  header.addEventListener('mouseleave', () => {
    header.style.background = '#f0f0f0';
  });

  const titleSpan = document.createElement('span');
  titleSpan.textContent = title;
  titleSpan.style.cssText = 'font-size: 12px; padding: 0 0 0 8px;';

  const indicator = document.createElement('span');
  indicator.textContent = '+';
  indicator.style.cssText = 'font-weight: bold; font-size: 16px;';

  header.appendChild(titleSpan);
  header.appendChild(indicator);

  const contentWrapper = document.createElement('div');
  contentWrapper.style.cssText = `
    display: none;
    padding: 12px;
    background: #f8f9fa;
    max-height: 800px;
    overflow-y: auto;
  `;
  contentWrapper.appendChild(content);

  section.appendChild(header);
  section.appendChild(contentWrapper);

  return { section, header, contentWrapper, indicator };
}

function createModelsSection(availableModels) {
  const container = document.createElement('div');

  const createSection = (labelText, element) => {
    const section = document.createElement('div');
    section.style.cssText = 'margin-bottom: 12px;';

    const label = document.createElement('label');
    label.textContent = labelText;
    label.style.cssText = `
      display: block;
      font-weight: 600;
      margin-bottom: 2px;
      color: #333;
    `;

    section.appendChild(label);
    section.appendChild(element);
    return section;
  };

  // Create Skip Glossary checkbox section
  const skipGlossarySection = document.createElement('div');
  skipGlossarySection.style.cssText = 'margin-bottom: 12px;';

  const skipGlossaryLabel = document.createElement('label');
  skipGlossaryLabel.style.cssText = `
    display: flex;
    align-items: center;
    font-weight: 600;
    color: #333;
    cursor: pointer;
  `;

  const skipGlossaryCheckbox = document.createElement('input');
  skipGlossaryCheckbox.type = 'checkbox';
  skipGlossaryCheckbox.id = 'skip-glossary-checkbox';
  skipGlossaryCheckbox.style.cssText = `
    all: revert;
    appearance: auto;
    margin-right: 6px;
  `;

  const skipGlossaryText = document.createTextNode('Skip Glossary Generation');

  skipGlossaryLabel.appendChild(skipGlossaryCheckbox);
  skipGlossaryLabel.appendChild(skipGlossaryText);
  skipGlossarySection.appendChild(skipGlossaryLabel);

  container.appendChild(skipGlossarySection);


  // Create model dropdown selectors
  // Create a container for both glossary selectors
  const glossaryContainer = document.createElement('div');
  glossaryContainer.id = 'glossary-selectors-container';

  const stage1Select = createModelSelector('stage1-select', availableModels, 'stage1');
  glossaryContainer.appendChild(createSection('Glossary Generate', stage1Select));

  const stage2Select = createModelSelector('stage2-select', availableModels, 'stage2');
  glossaryContainer.appendChild(createSection('Glossary Update', stage2Select));

  container.appendChild(glossaryContainer);

  const methodSelect = document.createElement('select');
  methodSelect.id = 'translation-method-select';
  methodSelect.style.cssText = stage1Select.style.cssText;

  const methods = [
    { value: 'single', label: 'Single (Line by Line)' },
    { value: 'chunk', label: 'Chunked (Smart Splitting)' },
    { value: 'entire', label: 'Entire (All at Once)' },
  ];

  for (const method of methods) {
    const option = document.createElement('option');
    option.value = method.value;
    option.textContent = method.label;
    methodSelect.appendChild(option);
  }
  container.appendChild(createSection('Translation Method', methodSelect));

  const stage3aSection = createSection('Text Chunking', createModelSelector('stage3a-select', availableModels), 'stage3a');
  stage3aSection.id = 'stage3a-section';
  container.appendChild(stage3aSection);

  const stage3bSelect = createModelSelector('stage3b-select', availableModels, 'stage3b');
  container.appendChild(createSection('Translation', stage3bSelect));

  return container;
}

function updateGlossarySelectorsVisibility(skipGlossary) {
  const glossaryContainer = document.getElementById('glossary-selectors-container');
  if (glossaryContainer) {
    glossaryContainer.style.display = skipGlossary ? 'none' : 'block';
  }
}

function createOptionsSection(availableModels) {
  const container = document.createElement('div');

  const createSection = (labelText, element) => {
    const section = document.createElement('div');
    section.style.cssText = 'margin-bottom: 12px;';

    const label = document.createElement('label');
    label.textContent = labelText;
    label.style.cssText = `
      display: block;
      font-weight: 600;
      margin-bottom: 2px;
      color: #333;
    `;

    section.appendChild(label);
    section.appendChild(element);
    return section;
  };

  // Get select styles from a reference selector
  const tempSelect = createModelSelector('temp-select', availableModels, '');
  const selectStyles = tempSelect.style.cssText;

  const contextInput = document.createElement('input');
  contextInput.type = 'number';
  contextInput.id = 'context-lines-input';
  contextInput.min = '1';
  contextInput.max = '10';
  contextInput.value = '3';
  contextInput.style.cssText = `
    width: 100%;
    padding: 4px;
    margin-top: 4px;
    border: 1px solid #ccc;
    border-radius: 3px;
    font-size: 12px;
  `;
  container.appendChild(createSection('Context Lines', contextInput));

  const narrativeSelect = document.createElement('select');
  narrativeSelect.id = 'narrative-select';
  narrativeSelect.style.cssText = selectStyles;

  const narratives = [
    { value: 'auto', label: 'Auto-detect' },
    { value: 'first', label: 'First Person' },
    { value: 'third', label: 'Third Person' },
  ];

  for (const narrative of narratives) {
    const option = document.createElement('option');
    option.value = narrative.value;
    option.textContent = narrative.label;
    narrativeSelect.appendChild(option);
  }
  container.appendChild(createSection('Preferred Narrative', narrativeSelect));

  const honorificsContainer = document.createElement('div');
  honorificsContainer.style.cssText = `
    margin-top: 4px;
    display: flex;
    align-items: center;
  `;

  const honorificsCheckbox = document.createElement('input');
  honorificsCheckbox.type = 'checkbox';
  honorificsCheckbox.id = 'honorifics-checkbox';
  honorificsCheckbox.style.cssText = `
    all: revert;
    appearance: auto;
    margin-right: 6px;
  `;

  const honorificsLabel = document.createElement('label');
  honorificsLabel.textContent = 'Preserve Japanese honorifics';
  honorificsLabel.htmlFor = 'honorifics-checkbox';
  honorificsLabel.style.cssText = 'cursor: pointer; user-select: none;';

  honorificsContainer.appendChild(honorificsCheckbox);
  honorificsContainer.appendChild(honorificsLabel);
  container.appendChild(createSection('Honorifics', honorificsContainer));

  const nameOrderSelect = document.createElement('select');
  nameOrderSelect.id = 'name-order-select';
  nameOrderSelect.style.cssText = selectStyles;

  const nameOrders = [
    { value: 'jp', label: 'Japanese (Family Name First)' },
    { value: 'en', label: 'Western (Given Name First)' },
  ];

  for (const order of nameOrders) {
    const option = document.createElement('option');
    option.value = order.value;
    option.textContent = order.label;
    nameOrderSelect.appendChild(option);
  }
  container.appendChild(createSection('Name Order', nameOrderSelect));

  const customInstructionsButton = document.createElement('button');
  customInstructionsButton.textContent = 'Edit Custom Instructions...';
  customInstructionsButton.style.cssText = `
    width: 100%;
    margin-top: 4px;
    padding: 6px;
    border: 1px solid #007bff;
    border-radius: 3px;
    background: white;
    color: #007bff;
    cursor: pointer;
    font-size: 12px;
  `;
  customInstructionsButton.addEventListener('mouseenter', () => {
    customInstructionsButton.style.background = '#007bff';
    customInstructionsButton.style.color = 'white';
  });
  customInstructionsButton.addEventListener('mouseleave', () => {
    customInstructionsButton.style.background = 'white';
    customInstructionsButton.style.color = '#007bff';
  });

  container.appendChild(createSection('Custom Instructions', customInstructionsButton));

  return container;
}

// Parent container for 'Models' and 'Options' sections combined
function createUIContainer(availableModels) {
  const parentContainer = document.createElement('div');
  parentContainer.style.cssText = `
    margin-top: 12px;
    font-size: 13px;
  `;

  const modelsContent = createModelsSection(availableModels);
  const optionsContent = createOptionsSection(availableModels);

  const modelsSection = createCollapsibleSection('Models', modelsContent);
  const optionsSection = createCollapsibleSection('Options', optionsContent);

  // Store sections for mutual exclusivity (easily extensible to 3+ sections)
  const sections = [modelsSection, optionsSection];

  const toggleSection = (targetSection) => {
    const isCurrentlyOpen = targetSection.contentWrapper.style.display !== 'none';

    // Close all sections first
    sections.forEach(s => {
      s.contentWrapper.style.display = 'none';
      s.indicator.textContent = '+';
    });

    // If the target was closed, open only it
    if (!isCurrentlyOpen) {
      targetSection.contentWrapper.style.display = 'block';
      targetSection.indicator.textContent = '−';
    }
  };

  modelsSection.header.addEventListener('click', () => toggleSection(modelsSection));
  optionsSection.header.addEventListener('click', () => toggleSection(optionsSection));

  parentContainer.appendChild(modelsSection.section);
  parentContainer.appendChild(optionsSection.section);

  return parentContainer;
}

/* ================================================================================
   UI State Persistance
   Save and load logic for settings to/from disk
   ================================================================================*/

async function loadUIState() {
  try {
    const savedState = await GM.getValue('translation_ui_state', null);
    if (!savedState) return null;

    // Validate that saved models are still available
    const availableModelIds = getAvailableModels().map(m => m.id);
    const validatedState = { ...savedState };

    // Check each model selection
    for (const key of ['stage1', 'stage2', 'stage3a', 'stage3b']) {
      if (validatedState[key] && !availableModelIds.includes(validatedState[key])) {
        console.warn(`[UI State] Saved model ${validatedState[key]} for ${key} is no longer available, resetting to null`);
        validatedState[key] = null;
      }
    }

    return validatedState;
  } catch (error) {
    console.error('[UI State] Failed to load state:', error);
    return null;
  }
}

async function saveUIState(state) {
  try {
    await GM.setValue('translation_ui_state', state);
  } catch (error) {
    console.error('[UI State] Failed to save state:', error);
  }
}

async function saveGlossaryBackup(stage1Value, stage2Value) {
  try {
    await GM.setValue('glossary_skip_backup', {
      stage1: stage1Value,
      stage2: stage2Value,
    });
  } catch (error) {
    console.error('[Glossary Skip] Failed to save backup:', error);
  }
}

async function loadGlossaryBackup() {
  try {
    return await GM.getValue('glossary_skip_backup', null);
  } catch (error) {
    console.error('[Glossary Skip] Failed to load backup:', error);
    return null;
  }
}

function getCurrentUIState() {
  const stage3aSelect = document.getElementById('stage3a-select');
  const translationMethod = document.getElementById('translation-method-select').value;
  const skipGlossary = document.getElementById('skip-glossary-checkbox').checked;

  return {
    stage1: skipGlossary ? null : (document.getElementById('stage1-select').value || null),
    stage2: skipGlossary ? null : (document.getElementById('stage2-select').value || null),
    stage3a: (translationMethod === 'chunk' && stage3aSelect) ? (stage3aSelect.value || null) : null,
    stage3b: document.getElementById('stage3b-select').value || null,
    translationMethod: translationMethod,
    contextLines: parseInt(document.getElementById('context-lines-input').value),
    narrative: document.getElementById('narrative-select').value,
    honorifics: document.getElementById('honorifics-checkbox').checked,
    nameOrder: document.getElementById('name-order-select').value,
    customInstructions: document.getElementById('custom-instruction-textarea').value.trim() || null,
    skipGlossary: skipGlossary,
  };
}

function applyUIState(state) {
  if (!state) return;

  const setValue = (id, value) => {
    const element = document.getElementById(id);
    if (element && value !== null && value !== undefined) {
      if (element.type === 'checkbox') {
        element.checked = value;
      } else {
        element.value = value;
      }
    }
  };

  setValue('skip-glossary-checkbox', state.skipGlossary || false);

  // If skip glossary is enabled, set stage1/stage2 to empty, otherwise apply saved values
  if (state.skipGlossary) {
    setValue('stage1-select', '');
    setValue('stage2-select', '');
  } else {
    setValue('stage1-select', state.stage1);
    setValue('stage2-select', state.stage2);
  }

  setValue('stage3a-select', state.stage3a);
  setValue('stage3b-select', state.stage3b);
  setValue('translation-method-select', state.translationMethod);
  setValue('context-lines-input', state.contextLines);
  setValue('narrative-select', state.narrative);
  setValue('honorifics-checkbox', state.honorifics);
  setValue('name-order-select', state.nameOrder);
  setValue('custom-instruction-textarea', state.customInstructions);

  // Trigger visibility update for stage3a
  updateStage3aVisibility(state.translationMethod);

  // Update glossary selectors visibility after other state is applied
  updateGlossarySelectorsVisibility(state.skipGlossary || false);
}

function updateStage3aVisibility(translationMethod) {
  const stage3aSection = document.getElementById('stage3a-section');
  if (stage3aSection) {
    stage3aSection.style.display = translationMethod === 'chunk' ? 'block' : 'none';
  }
}

function setupUIEventListeners() {
  // Skip glossary checkbox change
  const skipGlossaryCheckbox = document.getElementById('skip-glossary-checkbox');

  skipGlossaryCheckbox.addEventListener('change', async (e) => {
    const stage1Select = document.getElementById('stage1-select');
    const stage2Select = document.getElementById('stage2-select');
    const isChecked = e.target.checked;

    if (isChecked) {
      // Save current values before clearing
      const currentStage1 = stage1Select.value || null;
      const currentStage2 = stage2Select.value || null;
      await saveGlossaryBackup(currentStage1, currentStage2);

      // Clear the selectors
      stage1Select.value = '';
      stage2Select.value = '';
    } else {
      // Restore saved values
      const backup = await loadGlossaryBackup();
      if (backup) {
        stage1Select.value = backup.stage1 || '';
        stage2Select.value = backup.stage2 || '';
      }
    }

    // Update visibility
    updateGlossarySelectorsVisibility(isChecked);

    // Save UI state
    await saveUIState(getCurrentUIState());
  });

  // Translation method change
  const methodSelect = document.getElementById('translation-method-select');
  methodSelect.addEventListener('change', async (e) => {
    updateStage3aVisibility(e.target.value);
    await saveUIState(getCurrentUIState());
  });

  // Save state on any change
  const autoSaveElements = [
    'stage1-select', 'stage2-select', 'stage3a-select', 'stage3b-select',
    'context-lines-input', 'narrative-select', 'honorifics-checkbox',
    'name-order-select', 'custom-instruction-textarea',
  ];

  for (const id of autoSaveElements) {
    const element = document.getElementById(id);
    if (element) {
      element.addEventListener('change', async () => {
        await saveUIState(getCurrentUIState());
      });
    }
  }
}

async function initializeUI(floatingBox) {
  const availableModels = getAvailableModels();

  if (availableModels.length === 0) {
    const warning = document.createElement('div');
    warning.textContent = 'No API keys configured. Please set up at least one provider.';
    warning.style.cssText = `
      padding: 12px;
      background: #fff3cd;
      border: 1px solid #ffc107;
      border-radius: 4px;
      color: #856404;
      margin-top: 8px;
      font-size: 13px;
    `;
    floatingBox.appendChild(warning);
    return;
  }

  const uiContainer = createUIContainer(availableModels);
  floatingBox.appendChild(uiContainer);

  const customInstructionsDialog = createCustomInstructionsDialog();
  document.body.appendChild(customInstructionsDialog);

  // Hook up custom instructions button
  const customInstructionsButton = uiContainer.querySelector('button');
  customInstructionsButton.addEventListener('click', () => {
    customInstructionsDialog.style.display = 'flex';
  });

  // Load and apply saved state
  const savedState = await loadUIState();
  applyUIState(savedState);

  // Set up event listeners
  setupUIEventListeners();
}

// 'API Keys' / 'Dictionary' / 'Start translation' button
function createButton(label, onClick) {
  const button = document.createElement('button');
  button.textContent = label;
  button.style.cssText = `
    padding: 4px 16px;
    background: #4a90e2;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
    margin: 5px;
    transition: background 0.3s;
  `;

  button.addEventListener('mouseenter', () => {
    button.style.background = '#357abd';
  });

  button.addEventListener('mouseleave', () => {
    button.style.background = '#4a90e2';
  });

  button.addEventListener('click', onClick);

  return button;
}

function openApiKeysDialog() {
  // Prevent multiple dialogs
  if (document.getElementById('api-keys-dialog-overlay')) return;

  // Create overlay
  const overlay = document.createElement('div');
  overlay.id = 'api-keys-dialog-overlay';
  overlay.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 10000;
  `;

  // Create dialog
  const dialog = document.createElement('div');
  dialog.style.cssText = `
    background: white;
    border-radius: 8px;
    padding: 24px;
    min-width: 450px;
    max-width: 800px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    max-height: 80vh;
    overflow-y: auto;
  `;

  // Create title
  const title = document.createElement('h2');
  title.textContent = 'API Keys Configuration';
  title.style.cssText = `
    margin: 0 0 20px 0;
    color: #333;
    font-size: 20px;
    font-weight: 600;
  `;
  dialog.appendChild(title);

  // Create input fields for each provider
  const inputsContainer = document.createElement('div');
  const inputs = {};

  Object.keys(PROVIDER_API_CONFIG).forEach(provider => {
    const row = document.createElement('div');
    row.style.cssText = `
      margin-bottom: 16px;
    `;

    const label = document.createElement('label');
    label.textContent = provider.charAt(0).toUpperCase() + provider.slice(1);
    label.style.cssText = `
      display: block;
      margin-bottom: 6px;
      color: #555;
      font-size: 14px;
      font-weight: 500;
    `;

    const input = document.createElement('input');
    input.type = 'password';
    input.placeholder = `Enter ${provider} API key...`;
    input.value = PROVIDER_API_CONFIG[provider].apiKey || '';
    input.style.cssText = `
      width: 100%;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 4px;
      font-size: 14px;
      box-sizing: border-box;
      transition: border-color 0.3s;
    `;

    input.addEventListener('focus', () => {
      input.style.borderColor = '#4a90e2';
    });

    input.addEventListener('blur', () => {
      input.style.borderColor = '#ddd';
    });

    inputs[provider] = input;

    row.appendChild(label);
    row.appendChild(input);
    inputsContainer.appendChild(row);
  });

  dialog.appendChild(inputsContainer);

  // Create button container
  const buttonContainer = document.createElement('div');
  buttonContainer.style.cssText = `
    display: flex;
    justify-content: flex-end;
    gap: 10px;
    margin-top: 20px;
  `;

  // Create Close button
  const closeButton = document.createElement('button');
  closeButton.textContent = 'Close';
  closeButton.style.cssText = `
    padding: 10px 20px;
    background: #6c757d;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.3s;
  `;

  closeButton.addEventListener('mouseenter', () => {
    closeButton.style.background = '#5a6268';
  });

  closeButton.addEventListener('mouseleave', () => {
    closeButton.style.background = '#6c757d';
  });

  closeButton.addEventListener('click', () => {
    document.body.removeChild(overlay);
  });

  // Create Save button
  const saveButton = document.createElement('button');
  saveButton.textContent = 'Save';
  saveButton.style.cssText = `
    padding: 10px 20px;
    background: #28a745;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.3s;
  `;

  saveButton.addEventListener('mouseenter', () => {
    saveButton.style.background = '#218838';
  });

  saveButton.addEventListener('mouseleave', () => {
    saveButton.style.background = '#28a745';
  });

  saveButton.addEventListener('click', () => {
    const keys = {};
    Object.keys(inputs).forEach(provider => {
      const value = inputs[provider].value.trim();
      if (value) {
        keys[provider] = value;
      }
    });

    saveApiKeys(keys);

    // Optional: Show success feedback
    saveButton.textContent = 'Saved!';
    setTimeout(() => {
      saveButton.textContent = 'Save';
    }, 1500);
  });

  buttonContainer.appendChild(closeButton);
  buttonContainer.appendChild(saveButton);
  dialog.appendChild(buttonContainer);

  // Close on overlay click
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) {
      document.body.removeChild(overlay);
    }
  });

  overlay.appendChild(dialog);
  document.body.appendChild(overlay);
}

// Pop-up dialog box for inspection and editing of glossary / dictionary
function createDictionaryEditorDialog() {
  const domainManager = new DomainManager();

  // Add styles
  const style = document.createElement('style');
  style.textContent = `
    .dict-editor-overlay {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: rgba(0, 0, 0, 0.5);
      display: flex;
      justify-content: center;
      align-items: center;
      z-index: 10000;
    }

    .dict-editor-dialog {
      background: white;
      border-radius: 8px;
      width: 90%;
      max-width: 900px;
      max-height: 90vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }

    .dict-editor-header {
      padding: 20px;
      border-bottom: 2px solid #e0e0e0;
    }

    .dict-editor-header h2 {
      margin: 0 0 15px 0;
      font-size: 24px;
      color: #333;
    }

    .dict-editor-controls {
      display: flex;
      gap: 10px;
      align-items: center;
    }

    .dict-editor-search {
      flex: 1;
      display: flex;
      gap: 10px;
    }

    .dict-editor-search input {
      flex: 1;
      padding: 8px 12px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 14px;
    }

    .dict-editor-btn {
      padding: 8px 16px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 500;
      transition: background-color 0.2s;
    }

    .dict-editor-btn-primary {
      background: #007bff;
      color: white;
    }

    .dict-editor-btn-primary:hover {
      background: #0056b3;
    }

    .dict-editor-btn-secondary {
      background: #6c757d;
      color: white;
    }

    .dict-editor-btn-secondary:hover {
      background: #545b62;
    }

    .dict-editor-btn-success {
      background: #28a745;
      color: white;
    }

    .dict-editor-btn-success:hover {
      background: #218838;
    }

    .dict-editor-btn-danger {
      background: #dc3545;
      color: white;
    }

    .dict-editor-btn-danger:hover {
      background: #c82333;
    }

    .dict-editor-btn-small {
      padding: 4px 8px;
      font-size: 12px;
    }

    .dict-editor-content {
      flex: 1;
      overflow-y: auto;
      padding: 20px;
    }

    .dict-editor-entry {
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 15px;
      margin-bottom: 15px;
      background: #f9f9f9;
    }

    .dict-editor-entry.hidden {
      display: none;
    }

    .dict-editor-entry-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }

    .dict-editor-entry-title {
      font-weight: bold;
      color: #555;
    }

    .dict-editor-keys {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
      align-items: center;
    }

    .dict-editor-key-item {
      display: flex;
      gap: 5px;
      align-items: center;
    }

    .dict-editor-key-input {
      padding: 6px 10px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 14px;
      min-width: 150px;
    }

    .dict-editor-value {
      width: 100%;
      min-height: 80px;
      padding: 10px;
      border: 1px solid #ccc;
      border-radius: 4px;
      font-size: 14px;
      font-family: monospace;
      resize: vertical;
    }

    .dict-editor-footer {
      padding: 20px;
      border-top: 2px solid #e0e0e0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .dict-editor-footer-buttons {
      display: flex;
      gap: 10px;
    }
  `;
  document.head.appendChild(style);

  // Load dictionary
  let entries = loadDictionary(domainManager).entries || [];
  let currentEntries = [...entries];
  let sortOrder = null;

  // Create overlay
  const overlay = document.createElement('div');
  overlay.className = 'dict-editor-overlay';

  // Create dialog
  const dialog = document.createElement('div');
  dialog.className = 'dict-editor-dialog';

  // Build header
  const header = document.createElement('div');
  header.className = 'dict-editor-header';
  header.innerHTML = `
    <h2>Dictionary Editor</h2>
    <div class="dict-editor-controls">
      <div class="dict-editor-search">
        <input type="text" placeholder="Search keys or values..." class="dict-search-input">
        <button class="dict-editor-btn dict-editor-btn-primary dict-find-btn">Find</button>
      </div>
      <button class="dict-editor-btn dict-editor-btn-secondary dict-sort-btn">Sort</button>
      <button class="dict-editor-btn dict-editor-btn-secondary dict-reset-btn">Reset</button>
    </div>
  `;

  // Create content area
  const content = document.createElement('div');
  content.className = 'dict-editor-content';

  // Create footer
  const footer = document.createElement('div');
  footer.className = 'dict-editor-footer';
  footer.innerHTML = `
    <button class="dict-editor-btn dict-editor-btn-success dict-add-entry-btn">Add Entry</button>
    <div class="dict-editor-footer-buttons">
      <button class="dict-editor-btn dict-editor-btn-success dict-save-btn">Save Changes</button>
      <button class="dict-editor-btn dict-editor-btn-secondary dict-cancel-btn">Cancel</button>
    </div>
  `;

  dialog.appendChild(header);
  dialog.appendChild(content);
  dialog.appendChild(footer);
  overlay.appendChild(dialog);

  // Render entries
  const renderEntries = () => {
    content.innerHTML = '';
    currentEntries.forEach((entry, index) => {
      const entryDiv = document.createElement('div');
      entryDiv.className = 'dict-editor-entry';
      entryDiv.dataset.index = index.toString();

      const entryHeader = document.createElement('div');
      entryHeader.className = 'dict-editor-entry-header';

      const entryTitle = document.createElement('div');
      entryTitle.className = 'dict-editor-entry-title';
      entryTitle.textContent = `Entry #${index + 1}`;

      const deleteEntryBtn = document.createElement('button');
      deleteEntryBtn.className = 'dict-editor-btn dict-editor-btn-danger dict-editor-btn-small';
      deleteEntryBtn.textContent = 'Delete Entry';
      deleteEntryBtn.onclick = () => {
        currentEntries.splice(index, 1);
        renderEntries();
      };

      entryHeader.appendChild(entryTitle);
      entryHeader.appendChild(deleteEntryBtn);

      const keysDiv = document.createElement('div');
      keysDiv.className = 'dict-editor-keys';

      entry.keys.forEach((key, keyIndex) => {
        const keyItem = document.createElement('div');
        keyItem.className = 'dict-editor-key-item';

        const keyInput = document.createElement('input');
        keyInput.type = 'text';
        keyInput.className = 'dict-editor-key-input';
        keyInput.value = key;
        keyInput.oninput = (e) => {
          currentEntries[index].keys[keyIndex] = e.target.value;
        };

        const deleteKeyBtn = document.createElement('button');
        deleteKeyBtn.className = 'dict-editor-btn dict-editor-btn-danger dict-editor-btn-small';
        deleteKeyBtn.textContent = '✕';
        deleteKeyBtn.onclick = () => {
          currentEntries[index].keys.splice(keyIndex, 1);
          renderEntries();
        };

        keyItem.appendChild(keyInput);
        keyItem.appendChild(deleteKeyBtn);
        keysDiv.appendChild(keyItem);
      });

      const addKeyBtn = document.createElement('button');
      addKeyBtn.className = 'dict-editor-btn dict-editor-btn-primary dict-editor-btn-small';
      addKeyBtn.textContent = '+ Add Key';
      addKeyBtn.onclick = () => {
        currentEntries[index].keys.push('');
        renderEntries();
      };
      keysDiv.appendChild(addKeyBtn);

      const valueTextarea = document.createElement('textarea');
      valueTextarea.className = 'dict-editor-value';
      valueTextarea.value = entry.value;
      valueTextarea.oninput = (e) => {
        currentEntries[index].value = e.target.value;
      };

      entryDiv.appendChild(entryHeader);
      entryDiv.appendChild(keysDiv);
      entryDiv.appendChild(valueTextarea);
      content.appendChild(entryDiv);
    });
  };

  // Initial render
  renderEntries();

  // Event handlers
  const searchInput = header.querySelector('.dict-search-input');
  const findBtn = header.querySelector('.dict-find-btn');
  const sortBtn = header.querySelector('.dict-sort-btn');
  const resetBtn = header.querySelector('.dict-reset-btn');
  const addEntryBtn = footer.querySelector('.dict-add-entry-btn');
  const saveBtn = footer.querySelector('.dict-save-btn');
  const cancelBtn = footer.querySelector('.dict-cancel-btn');

  findBtn.onclick = () => {
    const searchTerm = searchInput.value.toLowerCase().trim();
    if (!searchTerm) {
      document.querySelectorAll('.dict-editor-entry').forEach(entry => {
        entry.classList.remove('hidden');
      });
      return;
    }

    document.querySelectorAll('.dict-editor-entry').forEach(entry => {
      const index = parseInt(entry.dataset.index);
      const entryData = currentEntries[index];
      const matchesKeys = entryData.keys.some(key =>
        key.toLowerCase().includes(searchTerm),
      );
      const matchesValue = entryData.value.toLowerCase().includes(searchTerm);

      if (matchesKeys || matchesValue) {
        entry.classList.remove('hidden');
      } else {
        entry.classList.add('hidden');
      }
    });
  };

  sortBtn.onclick = () => {
    if (sortOrder === 'desc') {
      currentEntries.sort((a, b) => a.value.length - b.value.length);
      sortOrder = 'asc';
    } else {
      currentEntries.sort((a, b) => b.value.length - a.value.length);
      sortOrder = 'desc';
    }
    renderEntries();
  };

  resetBtn.onclick = () => {
    currentEntries = [...entries];
    sortOrder = null;
    searchInput.value = '';
    renderEntries();
  };

  addEntryBtn.onclick = () => {
    currentEntries.push({ keys: [''], value: '' });
    renderEntries();
    content.scrollTop = content.scrollHeight;
  };

  saveBtn.onclick = () => {
    // Validate and warn about empty entries
    currentEntries.forEach((entry, index) => {
      if (entry.keys.length === 0 || entry.keys.every(k => !k.trim())) {
        console.warn(`Entry #${index + 1} has no valid keys`);
      }
      if (!entry.value.trim()) {
        console.warn(`Entry #${index + 1} has an empty value`);
      }
    });

    const dictionary = { entries: currentEntries };
    saveDictionary(domainManager, dictionary);
    document.body.removeChild(overlay);
  };

  cancelBtn.onclick = () => {
    document.body.removeChild(overlay);
  };

  overlay.onclick = (e) => {
    if (e.target === overlay) {
      document.body.removeChild(overlay);
    }
  };

  document.body.appendChild(overlay);
}

function createScrollableFloatingBox(topPos, leftPos) {
  // Wrapper handles positioning and scrolling
  const wrapper = document.createElement('div');
  wrapper.style.cssText = `
    position: fixed;
    top: ${topPos}px;
    left: ${leftPos}px;
    min-width: 200px;
    z-index: 1000;
    background-color: white;
    border: 1px solid #ccc;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
  `;

  // Inner content grows naturally
  const content = document.createElement('div');
  content.style.cssText = `
    padding: 25px 10px 10px 10px;
    display: flex;
    flex-direction: column;
    gap: 5px;
  `;

  wrapper.appendChild(content);

  const BOTTOM_MARGIN = 20; // Minimum clearance from viewport bottom

  function updateScrollability() {
    const maxAvailableHeight = window.innerHeight - topPos - BOTTOM_MARGIN;
    const contentHeight = content.scrollHeight;

    if (contentHeight > maxAvailableHeight) {
      wrapper.style.maxHeight = `${maxAvailableHeight}px`;
      wrapper.style.overflowY = 'auto';
    } else {
      wrapper.style.maxHeight = 'none';
      wrapper.style.overflowY = 'visible';
    }
  }

  // Watch for content size changes
  const resizeObserver = new ResizeObserver(updateScrollability);
  resizeObserver.observe(content);

  // Handle window resize
  window.addEventListener('resize', updateScrollability);

  // Initial check after DOM settles
  setTimeout(updateScrollability, 0);

  return { wrapper, content };
}

// Custom CSS for translated text
function addStyles() {
  if (document.getElementById(`${SCRIPT_PREFIX}-style`)) return;
  const style = document.createElement('style');
  style.id = `${SCRIPT_PREFIX}-style`;
  style.textContent = `
    .${SCRIPT_PREFIX}text {
      font-family: "Lucida Sans", serif;
      text-indent: 1.5em;
      padding: 5px 0 5px 0;
    }
    .${SCRIPT_PREFIX}text span {
      display: block;
    }
  `;
  document.head.appendChild(style);
}

/* ================================================================================
   Error Handling
   ================================================================================*/

let errorCount = 0;

function errorHandler() {
  const display = document.getElementById('error-display');

  errorCount++;

  // Un-hide it on the first error
  if (errorCount === 1) {
    display.style.display = 'block';
  }

  display.textContent = `Something went wrong, check console for details. Errors: ${errorCount}`;
}

/* ================================================================================
   Init
   ================================================================================*/

function init() {
  // Load from storage
  loadApiKeys();

  // Button to collapse floating box UI
  const collapseButton = document.createElement('button');
  collapseButton.textContent = '▼';
  collapseButton.id = 'userscript-collapseButton';
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
  miniButton.textContent = '▲';
  document.body.appendChild(miniButton);

  // Create scrollable floating box structure
  const { wrapper: floatingBoxWrapper, content: floatingBox } = createScrollableFloatingBox(60, 3);

  // Add the collapse button to floating box
  floatingBox.appendChild(collapseButton);
  document.body.appendChild(floatingBoxWrapper);

  // Toggle function - now operates on the wrapper
  function toggleUI() {
    if (floatingBoxWrapper.style.display !== 'none') {
      floatingBoxWrapper.style.display = 'none';
      miniButton.style.display = 'flex';
    } else {
      floatingBoxWrapper.style.display = 'block';
      miniButton.style.display = 'none';
    }
  }

  collapseButton.addEventListener('click', toggleUI);
  miniButton.addEventListener('click', toggleUI);

  // Title
  const title = document.createElement('h2');
  title.textContent = 'Translation Control';
  title.style.cssText = `
    margin: 0 0 0 0;
    padding: 0 0 0 0;
    font-size: 14px;
    color: #333;
  `;
  floatingBox.appendChild(title);

  // Error display
  const errorDisplay = document.createElement('div');
  errorDisplay.id = 'error-display';
  errorDisplay.style.cssText = `
    display: none;
    padding: 8px;
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 4px;
    color: #721c24;
    margin-bottom: 12px;
    font-size: 12px;
  `;
  floatingBox.appendChild(errorDisplay);

  // ------------------------------
  // API keys dialog box
  const apiKeysButton = createButton('API Keys', openApiKeysDialog);
  floatingBox.appendChild(apiKeysButton);

  // ------------------------------
  // Metadata edit dialog box
  const openEditorBtn = createButton('Dictionary Editor', createDictionaryEditorDialog)
  floatingBox.appendChild(openEditorBtn);

  // ------------------------------
  // Start translation button
  const translateButton = createButton('Start Translation', () => {
    errorDisplay.style.display = 'none';

    const validationResult = UIConfigValidator.validate({ modelsList });
    if (validationResult === null) {
      errorDisplay.textContent = 'Configuration validation failed. Please check your settings.';
      errorDisplay.style.display = 'block';
      return;
    }

    translateButton.disabled = true; // Should only be needed to be run once per page, for most sites.
    translateButton.textContent = 'Translating...';

    main()
      .then(() => {
        translateButton.textContent = 'Translation Complete!';
        const collaspeButton = document.getElementById('userscript-collapseButton');
        collaspeButton.click(); // Hide UI once done
      })
      .catch(error => {
        console.error('[Translation] Error:', error);
        errorDisplay.textContent = `Translation failed: ${error.message}`;
        errorDisplay.style.display = 'block';
        translateButton.disabled = false;
        translateButton.textContent = 'Start Translation';
      });
  })
  floatingBox.appendChild(translateButton);

  // ------------------------------
  // Progress display
  const progressSection = document.createElement('div');
  progressSection.id = 'progress-section';
  progressSection.style.marginTop = '8px';
  floatingBox.appendChild(progressSection);

  void initializeUI(floatingBox);

  addStyles();
}

window.addEventListener('load', init);

/* ================================================================================
    Test
   ================================================================================*/

if (TEST_MODE) {
  // Add test models to config
  PROVIDER_API_CONFIG.test = {
    apiKey: 'example-api-key',
    endpoint: 'https://api.example.test/v1/chat/completions',
  }

  PROVIDER_CONFIGS.test = {
    models: [
      { id: '99-1', model: 'test-model-1', label: 'Test Model #1' },
      { id: '99-2', model: 'test-model-2', label: 'Test Model #2' },
      { id: '99-3', model: 'test-model-3', label: 'Test Model #3' },
      { id: '99-4', model: 'test-model-4', label: 'Test Model #4' },
    ],
    limits: {
      stage1: 'all',
      stage2: 'all',
      stage3a: 'all',
      stage3b: 'all',
    },
  }
}

// Test counter for mock responses
let TEST_RESPONSE_COUNTER = 0;

/**
 * Mock response generator for test provider.
 * Bypasses actual HTTP calls and returns simulated LLM responses.
 */
function generateTestResponse(modelId, messages) {
  const extractOriginalRaw = (userMessage) => {
    return userMessage.split('Translate the following Japanese text into English:')[1];
  }

  // Generate random intervals
  const simChunkerResponse = (userMessage) => {
    // Helper random fake internal genrator
    const generateIntervals = (s, e) => {
      const X = 5;
      const Y = 12;
      const intervals = [];
      let start = s;

      while (start <= e) {
        // Random length between X and Y
        const length = Math.floor(Math.random() * (Y - X + 1)) + X;
        let end = start + length - 1;

        // If this would exceed N, just make the last interval end at N
        if (end >= e) {
          end = e;
          intervals.push([start, end]);
          break;
        }

        intervals.push([start, end]);
        start = end + 1;
      }

      // Format with spaces: [1, 5] instead of [1,5]
      const formatted = intervals.map(([a, b]) => `[${a}, ${b}]`).join(', ');
      return `[${formatted}]`;
    }

    // Extract the starting and ending indices from prompt
    const lines = userMessage.split('\n');
    let X = null;
    let Y = null;

    for (const line of lines) {
      if (line.startsWith('Start:')) {
        X = parseInt(line.split(':')[1].trim());
      } else if (line.startsWith('End:')) {
        Y = parseInt(line.split(':')[1].trim());
      }
    }

    return generateIntervals(X, Y);
  }

  const userMessage = messages.find(m => m.role === 'user')?.content || '';

  if (modelId === 'test-model-1') {
    return {
      choices: [{
        message: {
          content: `
\`\`\`json
{
  "entries": [
    {
      "id": 1,
      "keys": ["名無しの権兵衛", "ななしのごんべい"],
      "value": "[character] Name: John Doe (名無しの権兵衛) | Gender: Male | Nickname: Nanashi (ななし)"
    },
    {
      "id": 2,
      "keys": ["アメリカ合衆国", "アメリカ"],
      "value": "[location] Name: United States (アメリカ)"
    }
  ]
}
\`\`\`
`.trim(),
        },
      }],
    };
  }

  if (modelId === 'test-model-2') {
    return {
      choices: [{
        message: {
          content: `{ "action": "none" }`,
        },
      }],
    };
  }

  if (modelId === 'test-model-3') {
    return {
      choices: [{
        message: {
          content: `${simChunkerResponse(userMessage)}`,
        },
      }],
    };
  }

  if (modelId === 'test-model-4') {
    TEST_RESPONSE_COUNTER++;
    return {
      choices: [{
        message: {
          content: `<translation>Test Translation #${TEST_RESPONSE_COUNTER}, Echo: ${extractOriginalRaw(userMessage)}</translation>`,
        },
      }],
    };
  }

  // Default test response
  return {
    choices: [{
      message: {
        content: `Model ${modelId} default resposne.`,
      },
    }],
  };
}
