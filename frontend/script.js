// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const extractBtn = document.getElementById('extractBtn');
const promptInput = document.getElementById('promptInput');
const sessionIdInput = document.getElementById('sessionId');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const summaryCards = document.getElementById('summaryCards');
const flowContainer = document.getElementById('flowContainer');
const viewBehaviorsBtn = document.getElementById('viewBehaviorsBtn');
const viewConflictsBtn = document.getElementById('viewConflictsBtn');
const behaviorsModal = document.getElementById('behaviorsModal');
const conflictsModal = document.getElementById('conflictsModal');
const closeBehaviorsModal = document.getElementById('closeBehaviorsModal');
const closeConflictsModal = document.getElementById('closeConflictsModal');
const behaviorsModalBody = document.getElementById('behaviorsModalBody');
const conflictsModalBody = document.getElementById('conflictsModalBody');

// Store current session ID
let currentSessionId = '';

// Event Listeners
extractBtn.addEventListener('click', handleExtraction);
viewBehaviorsBtn.addEventListener('click', showAllBehaviors);
viewConflictsBtn.addEventListener('click', showAllConflicts);
closeBehaviorsModal.addEventListener('click', () => behaviorsModal.style.display = 'none');
closeConflictsModal.addEventListener('click', () => conflictsModal.style.display = 'none');

// Close modals when clicking outside
window.addEventListener('click', (e) => {
    if (e.target === behaviorsModal) {
        behaviorsModal.style.display = 'none';
    }
    if (e.target === conflictsModal) {
        conflictsModal.style.display = 'none';
    }
});

// Main extraction handler
async function handleExtraction() {
    const prompt = promptInput.value.trim();
    const sessionId = sessionIdInput.value.trim() || 'default';

    if (!prompt) {
        showError('Please enter a prompt');
        return;
    }

    currentSessionId = sessionId;
    setLoading(true);
    hideError();
    hideResults();

    try {
        const response = await fetch(`${API_BASE_URL}/extract-detailed`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                prompt: prompt,
                session_id: sessionId
            })
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Extraction failed');
        }

        displayResults(data.data);
    } catch (error) {
        showError(`Error: ${error.message}`);
    } finally {
        setLoading(false);
    }
}

// Display results
function displayResults(data) {
    hideError();
    resultsSection.style.display = 'block';

    // Display summary cards
    displaySummaryCards(data.processing);

    // Display behavior flow
    displayBehaviorFlow(data.flow_info);
}

// Display summary cards
function displaySummaryCards(processing) {
    summaryCards.innerHTML = `
        <div class="summary-card card-extracted">
            <div class="card-icon">📝</div>
            <div class="card-content">
                <div class="card-number">${processing.total_extracted}</div>
                <div class="card-label">Extracted</div>
            </div>
        </div>
        <div class="summary-card card-stored">
            <div class="card-icon">💾</div>
            <div class="card-content">
                <div class="card-number">${processing.total_stored}</div>
                <div class="card-label">Stored</div>
            </div>
        </div>
        <div class="summary-card card-reinforced">
            <div class="card-icon">🔁</div>
            <div class="card-content">
                <div class="card-number">${processing.total_reinforced}</div>
                <div class="card-label">Reinforced</div>
            </div>
        </div>
        <div class="summary-card card-conflicts">
            <div class="card-icon">⚠️</div>
            <div class="card-content">
                <div class="card-number">${processing.total_conflicts}</div>
                <div class="card-label">Conflicts</div>
            </div>
        </div>
        <div class="summary-card card-pruned">
            <div class="card-icon">✂️</div>
            <div class="card-content">
                <div class="card-number">${processing.total_pruned}</div>
                <div class="card-label">Pruned</div>
            </div>
        </div>
    `;
}

// Display behavior flow
function displayBehaviorFlow(flowInfo) {
    if (!flowInfo || flowInfo.length === 0) {
        flowContainer.innerHTML = '<p class="no-data">No behaviors processed</p>';
        return;
    }

    flowContainer.innerHTML = flowInfo.map((flow, index) => {
        return createFlowCard(flow, index + 1);
    }).join('');
}

// Create a flow card for a single behavior
function createFlowCard(flow, index) {
    const actionClass = getActionClass(flow.action);
    const actionIcon = getActionIcon(flow.action);
    const actionLabel = getActionLabel(flow.action);

    return `
        <div class="flow-card ${actionClass}">
            <div class="flow-header">
                <span class="flow-number">#${index}</span>
                <span class="flow-action">${actionIcon} ${actionLabel}</span>
            </div>
            <div class="flow-body">
                <div class="behavior-text">
                    <strong>Behavior:</strong> ${escapeHtml(flow.behavior_description)}
                </div>
                ${createCanonicalInfo(flow.canonical)}
                <div class="flow-metrics">
                    <span class="metric">
                        <strong>Credibility:</strong> ${flow.credibility.toFixed(3)}
                    </span>
                    ${flow.distance !== null ? `
                        <span class="metric">
                            <strong>Distance:</strong> ${flow.distance.toFixed(4)}
                        </span>
                    ` : ''}
                </div>
                ${createMatchedBehaviorInfo(flow)}
                ${createConflictInfo(flow)}
                ${flow.details ? `
                    <div class="flow-details">
                        <strong>Details:</strong> ${escapeHtml(flow.details)}
                    </div>
                ` : ''}
                ${flow.stored_behavior_id ? `
                    <div class="stored-id">
                        <strong>Stored ID:</strong> <code>${flow.stored_behavior_id}</code>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}

// Create canonical information display
function createCanonicalInfo(canonical) {
    if (!canonical) {
        return '<div class="canonical-info missing">⚠️ Canonical fields missing</div>';
    }

    const polarityClass = canonical.polarity === 'POSITIVE' ? 'positive' : 'negative';
    const polarityIcon = canonical.polarity === 'POSITIVE' ? '👍' : '👎';

    return `
        <div class="canonical-info">
            <div class="canonical-grid">
                <div class="canonical-item">
                    <span class="canonical-label">Intent:</span>
                    <span class="canonical-value intent">${canonical.intent}</span>
                </div>
                <div class="canonical-item">
                    <span class="canonical-label">Target:</span>
                    <span class="canonical-value target">${escapeHtml(canonical.target)}</span>
                </div>
                <div class="canonical-item">
                    <span class="canonical-label">Context:</span>
                    <span class="canonical-value context">${escapeHtml(canonical.context)}</span>
                </div>
                <div class="canonical-item">
                    <span class="canonical-label">Polarity:</span>
                    <span class="canonical-value polarity ${polarityClass}">
                        ${polarityIcon} ${canonical.polarity}
                    </span>
                </div>
            </div>
        </div>
    `;
}

// Create matched behavior information
function createMatchedBehaviorInfo(flow) {
    if (!flow.matched_behavior_id) {
        return '';
    }

    return `
        <div class="matched-behavior">
            <strong>Matched Behavior:</strong>
            <div class="matched-content">
                <div><strong>ID:</strong> <code>${flow.matched_behavior_id}</code></div>
                <div><strong>Text:</strong> ${escapeHtml(flow.matched_behavior_text)}</div>
            </div>
        </div>
    `;
}

// Create conflict information display
function createConflictInfo(flow) {
    if (!flow.conflict_info) {
        return '';
    }

    const conflict = flow.conflict_info;
    let conflictHtml = '<div class="conflict-info">';
    conflictHtml += '<strong>⚠️ Conflict Details:</strong>';
    conflictHtml += '<div class="conflict-content">';

    if (conflict.conflict_type) {
        conflictHtml += `<div><strong>Type:</strong> ${conflict.conflict_type}</div>`;
    }

    if (conflict.existing_polarity && conflict.new_polarity) {
        conflictHtml += `
            <div class="polarity-conflict">
                <strong>Polarity Mismatch:</strong> 
                ${conflict.existing_polarity} → ${conflict.new_polarity}
            </div>
        `;
    }

    if (conflict.resolution) {
        conflictHtml += `<div><strong>Resolution:</strong> ${conflict.resolution}</div>`;
    }

    if (conflict.explanation) {
        conflictHtml += `<div><strong>Explanation:</strong> ${escapeHtml(conflict.explanation)}</div>`;
    }

    if (conflict.llm_analysis) {
        conflictHtml += `
            <div class="llm-analysis">
                <strong>LLM Analysis:</strong> 
                <p>${escapeHtml(conflict.llm_analysis)}</p>
                ${conflict.llm_confidence ? `
                    <span class="confidence">Confidence: ${conflict.llm_confidence.toFixed(2)}</span>
                ` : ''}
            </div>
        `;
    }

    conflictHtml += '</div></div>';
    return conflictHtml;
}

// Get action-specific CSS class
function getActionClass(action) {
    const classMap = {
        'NEW_BEHAVIOR': 'action-new',
        'DUPLICATE_REINFORCED': 'action-duplicate',
        'CONFLICT_DETECTED': 'action-conflict',
        'CONFLICT_AUTO_RESOLVED': 'action-conflict',
        'SUPERSEDED_EXISTING': 'action-superseded',
        'IGNORED_NEW': 'action-ignored',
        'COMPATIBLE': 'action-compatible',
        'PRUNED': 'action-pruned'
    };
    return classMap[action] || 'action-default';
}

// Get action icon
function getActionIcon(action) {
    const iconMap = {
        'NEW_BEHAVIOR': '✨',
        'DUPLICATE_REINFORCED': '🔁',
        'CONFLICT_DETECTED': '⚠️',
        'CONFLICT_AUTO_RESOLVED': '✅',
        'SUPERSEDED_EXISTING': '🔄',
        'IGNORED_NEW': '🚫',
        'COMPATIBLE': '✔️',
        'PRUNED': '✂️'
    };
    return iconMap[action] || '•';
}

// Get action label
function getActionLabel(action) {
    const labelMap = {
        'NEW_BEHAVIOR': 'New Behavior',
        'DUPLICATE_REINFORCED': 'Duplicate - Reinforced',
        'CONFLICT_DETECTED': 'Conflict Detected',
        'CONFLICT_AUTO_RESOLVED': 'Conflict Auto-Resolved',
        'SUPERSEDED_EXISTING': 'Superseded Existing',
        'IGNORED_NEW': 'Ignored (Lower Credibility)',
        'COMPATIBLE': 'Compatible',
        'PRUNED': 'Pruned (Low Credibility)'
    };
    return labelMap[action] || action;
}

// Show all behaviors modal
async function showAllBehaviors() {
    try {
        behaviorsModalBody.innerHTML = '<div class="loading">Loading behaviors...</div>';
        behaviorsModal.style.display = 'block';

        const response = await fetch(`${API_BASE_URL}/behaviors/${currentSessionId}`);
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Failed to fetch behaviors');
        }

        displayBehaviorsModal(data.data.behaviors);
    } catch (error) {
        behaviorsModalBody.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

// Display behaviors in modal
function displayBehaviorsModal(behaviors) {
    if (!behaviors || behaviors.length === 0) {
        behaviorsModalBody.innerHTML = '<p class="no-data">No behaviors found</p>';
        return;
    }

    behaviorsModalBody.innerHTML = behaviors.map((behavior) => {
        const stateClass = behavior.behavior_state.toLowerCase();
        const polarityClass = behavior.polarity === 'POSITIVE' ? 'positive' : 'negative';
        const polarityIcon = behavior.polarity === 'POSITIVE' ? '👍' : '👎';

        return `
            <div class="behavior-card state-${stateClass}">
                <div class="behavior-header">
                    <span class="behavior-id"><code>${behavior.behavior_id}</code></span>
                    <span class="behavior-state">${behavior.behavior_state}</span>
                </div>
                <div class="behavior-content">
                    <div class="behavior-text-display">
                        ${escapeHtml(behavior.behavior_text)}
                    </div>
                    <div class="canonical-grid">
                        <div class="canonical-item">
                            <span class="canonical-label">Intent:</span>
                            <span class="canonical-value intent">${behavior.intent}</span>
                        </div>
                        <div class="canonical-item">
                            <span class="canonical-label">Target:</span>
                            <span class="canonical-value target">${escapeHtml(behavior.target)}</span>
                        </div>
                        <div class="canonical-item">
                            <span class="canonical-label">Context:</span>
                            <span class="canonical-value context">${escapeHtml(behavior.context)}</span>
                        </div>
                        <div class="canonical-item">
                            <span class="canonical-label">Polarity:</span>
                            <span class="canonical-value polarity ${polarityClass}">
                                ${polarityIcon} ${behavior.polarity}
                            </span>
                        </div>
                    </div>
                    <div class="behavior-metrics">
                        <span class="metric">
                            <strong>Credibility:</strong> ${behavior.credibility.toFixed(3)}
                        </span>
                        <span class="metric">
                            <strong>Reinforcements:</strong> ${behavior.reinforcement_count}
                        </span>
                        <span class="metric">
                            <strong>Clarity:</strong> ${behavior.clarity_score.toFixed(2)}
                        </span>
                        <span class="metric">
                            <strong>Confidence:</strong> ${behavior.extraction_confidence.toFixed(2)}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Show all conflicts modal
async function showAllConflicts() {
    try {
        conflictsModalBody.innerHTML = '<div class="loading">Loading conflicts...</div>';
        conflictsModal.style.display = 'block';

        const response = await fetch(`${API_BASE_URL}/conflicts/${currentSessionId}`);
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Failed to fetch conflicts');
        }

        displayConflictsModal(data.data.conflicts);
    } catch (error) {
        conflictsModalBody.innerHTML = `<div class="error">Error: ${error.message}</div>`;
    }
}

// Display conflicts in modal
function displayConflictsModal(conflicts) {
    if (!conflicts || conflicts.length === 0) {
        conflictsModalBody.innerHTML = '<p class="no-data">No conflicts found</p>';
        return;
    }

    conflictsModalBody.innerHTML = conflicts.map((conflict) => {
        const statusClass = conflict.resolution_status.toLowerCase();

        return `
            <div class="conflict-card status-${statusClass}">
                <div class="conflict-header">
                    <span class="conflict-id"><code>${conflict.conflict_id}</code></span>
                    <span class="conflict-status">${conflict.resolution_status}</span>
                </div>
                <div class="conflict-content">
                    <div class="conflict-type">
                        <strong>Type:</strong> ${conflict.conflict_type}
                    </div>
                    <div class="conflict-behaviors">
                        <div class="conflict-behavior">
                            <strong>Behavior 1:</strong>
                            <div class="conflict-behavior-details">
                                <div><code>${conflict.behavior_id_1}</code></div>
                                <div class="behavior-text-display">${escapeHtml(conflict.behavior_1_text)}</div>
                                <div>Credibility: ${conflict.behavior_1_credibility.toFixed(3)} | State: ${conflict.behavior_1_state}</div>
                            </div>
                        </div>
                        <div class="conflict-vs">⚔️</div>
                        <div class="conflict-behavior">
                            <strong>Behavior 2:</strong>
                            <div class="conflict-behavior-details">
                                <div><code>${conflict.behavior_id_2}</code></div>
                                <div class="behavior-text-display">${escapeHtml(conflict.behavior_2_text)}</div>
                                <div>Credibility: ${conflict.behavior_2_credibility.toFixed(3)} | State: ${conflict.behavior_2_state}</div>
                            </div>
                        </div>
                    </div>
                    ${conflict.llm_analysis ? `
                        <div class="llm-analysis">
                            <strong>LLM Analysis:</strong>
                            <p>${escapeHtml(conflict.llm_analysis)}</p>
                        </div>
                    ` : ''}
                    <div class="conflict-metrics">
                        <span class="metric">
                            <strong>Distance:</strong> ${conflict.similarity_distance.toFixed(4)}
                        </span>
                        <span class="metric">
                            <strong>Created:</strong> ${new Date(conflict.created_at * 1000).toLocaleString()}
                        </span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

// Utility functions
function setLoading(isLoading) {
    const btnText = extractBtn.querySelector('.btn-text');
    const loader = extractBtn.querySelector('.loader');

    extractBtn.disabled = isLoading;
    btnText.style.display = isLoading ? 'none' : 'inline';
    loader.style.display = isLoading ? 'inline-block' : 'none';
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    resultsSection.style.display = 'none';
}

function hideError() {
    errorSection.style.display = 'none';
}

function hideResults() {
    resultsSection.style.display = 'none';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
