import { API_BASE_URL } from '../constants';
import { ExtractRequest, ExtractResponse, BehaviorsResponse, ConflictsResponse } from '../types';

export const extractDetailed = async (
  payload: ExtractRequest
): Promise<ExtractResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/extract-detailed`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    const data: ExtractResponse = await response.json();

    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to extract behaviors');
    }

    return data;
  } catch (error) {
    console.error('API Error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown network error',
      message: 'Network request failed'
    };
  }
};

export const getBehaviors = async (sessionId: string): Promise<BehaviorsResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/behaviors/${sessionId}`);
    const data: BehaviorsResponse = await response.json();

    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to fetch behaviors');
    }
    return data;
  } catch (error) {
    console.error('API Error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown network error',
      message: 'Network request failed'
    };
  }
};

export const getConflicts = async (sessionId: string): Promise<ConflictsResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/conflicts/${sessionId}`);
    const data: ConflictsResponse = await response.json();

    if (!response.ok) {
      throw new Error(data.message || data.error || 'Failed to fetch conflicts');
    }
    return data;
  } catch (error) {
    console.error('API Error:', error);
    return {
      success: false,
      error: error instanceof Error ? error.message : 'Unknown network error',
      message: 'Network request failed'
    };
  }
};
