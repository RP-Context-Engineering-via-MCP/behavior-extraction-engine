// ============================================================================
// TYPE DEFINITIONS FROM API SPEC
// ============================================================================

export type ActionType = 
  | 'NEW_BEHAVIOR'
  | 'DUPLICATE_REINFORCED'
  | 'CONFLICT_DETECTED'
  | 'CONFLICT_AUTO_RESOLVED'
  | 'CONFLICT_LLM_RESOLVED'
  | 'SUPERSEDED_EXISTING'
  | 'IGNORED_NEW'
  | 'COMPATIBLE'
  | 'PRUNED';

export type ConflictType = 
  | 'POLARITY_MISMATCH'
  | 'INTENT_CONFLICT'
  | 'SEMANTIC_CONFLICT'
  | 'CONTEXT_CONFLICT';

export type ConflictResolution = 
  | 'SUPERSEDE_EXISTING'
  | 'IGNORE_NEW'
  | 'KEEP_BOTH'
  | 'PENDING_USER'
  | 'AUTO_RESOLVED'
  | 'LLM_RESOLVED';

export type ResolutionStatus = 
  | 'RESOLVED'
  | 'PENDING'
  | 'AUTO_RESOLVED'
  | 'USER_RESOLVED';

export type BehaviorState = 
  | 'ACTIVE'
  | 'SUPERSEDED'
  | 'INACTIVE'
  | 'PENDING_RESOLUTION';

export type Polarity = 'POSITIVE' | 'NEGATIVE';

export type ResolutionMethod = 'AUTO' | 'LLM' | 'MANUAL';

// -------------------- Core Data Structures --------------------

export interface CanonicalFields {
  intent: string;
  target: string;
  context: string;
  polarity: Polarity;
}

export interface ReinforcementInfo {
  reinforcement_count: number;
  previous_credibility: number;
  new_credibility: number;
  credibility_increase: number;
  reinforcement_factor: number;
  total_reinforcements: number;
  message: string;
}

export interface ConflictInfo {
  conflict_id: string;
  conflict_type: ConflictType;
  conflicting_behavior_id: string;
  conflicting_behavior_text: string;
  conflicting_behavior_credibility: number;
  conflicting_behavior_state: BehaviorState;
  existing_polarity?: Polarity;
  new_polarity?: Polarity;
  polarity_mismatch?: boolean;
  resolution: ConflictResolution;
  resolution_status: ResolutionStatus;
  resolution_method?: ResolutionMethod;
  resolution_reason?: string;
  llm_analysis?: string;
  llm_confidence?: number;
  llm_recommendation?: string;
  winner_behavior_id?: string;
  loser_behavior_id?: string;
  action_taken?: string;
  similarity_distance: number;
  credibility_difference: number;
  requires_user_resolution: boolean;
  user_resolution_message?: string;
  resolution_options?: string[];
  explanation?: string;
  detected_at: number;
  resolved_at?: number;
}

export interface BehaviorFlow {
  behavior_id: string | null;
  behavior_description: string;
  behavior_text: string;
  canonical: CanonicalFields | null;
  action: ActionType;
  credibility: number;
  initial_credibility?: number;
  credibility_delta?: number;
  clarity_score: number;
  extraction_confidence: number;
  distance: number | null;
  similarity_score?: number;
  similarity_threshold_used: number;
  matched_behavior_id: string | null;
  matched_behavior_text: string | null;
  matched_behavior_credibility?: number;
  matched_behavior_state?: BehaviorState;
  reinforcement_info?: ReinforcementInfo;
  conflict_info?: ConflictInfo;
  stored_behavior_id: string | null;
  behavior_state?: BehaviorState;
  details?: string;
  processing_notes?: string;
  timestamp: number;
}

export interface ProcessingSummary {
  total_extracted: number;
  total_stored: number;
  total_reinforced: number;
  total_conflicts: number;
  total_pruned: number;
  total_auto_resolved: number;
  total_llm_resolved: number;
}

// -------------------- Stored Data Types --------------------

export interface StoredBehavior {
  behavior_id: string;
  behavior_text: string;
  behavior_state: BehaviorState;
  intent: string;
  target: string;
  context: string;
  polarity: Polarity;
  credibility: number;
  reinforcement_count: number;
  clarity_score: number;
  extraction_confidence: number;
}

export interface StoredConflict {
  conflict_id: string;
  resolution_status: ResolutionStatus;
  conflict_type: ConflictType;
  behavior_id_1: string;
  behavior_1_text: string;
  behavior_1_credibility: number;
  behavior_1_state: BehaviorState;
  behavior_id_2: string;
  behavior_2_text: string;
  behavior_2_credibility: number;
  behavior_2_state: BehaviorState;
  llm_analysis?: string;
  similarity_distance: number;
  created_at: number;
}

export interface BehaviorsResponseData {
  behaviors: StoredBehavior[];
}

export interface ConflictsResponseData {
  conflicts: StoredConflict[];
}

// -------------------- API Request Types --------------------

export interface ExtractRequest {
  prompt: string;
  session_id: string;
  enable_llm_conflict_resolution?: boolean;
}

// -------------------- API Response Types --------------------

export interface ApiResponse<T> {
  success: boolean;
  message?: string;
  data?: T;
  error?: string;
  error_code?: string;
  details?: string;
}

export interface ExtractResponseData {
  processing: ProcessingSummary;
  flow_info: BehaviorFlow[];
  session_id: string;
  extraction_timestamp: number;
}

export type ExtractResponse = ApiResponse<ExtractResponseData>;
export type BehaviorsResponse = ApiResponse<BehaviorsResponseData>;
export type ConflictsResponse = ApiResponse<ConflictsResponseData>;