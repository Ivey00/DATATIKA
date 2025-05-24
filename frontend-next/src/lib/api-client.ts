/**
 * API client for making requests to the backend
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ApiRequestOptions extends RequestInit {
  method?: string;
  headers?: HeadersInit;
  body?: any;
}

interface ApiResponse<T = any> {
  success: boolean;
  message: string;
  data?: T;
  [key: string]: any;
}

/**
 * Makes an API request to the backend
 * @param endpoint - The API endpoint to call (without leading slash)
 * @param options - Request options
 * @returns Promise with the API response
 */
export async function apiRequest<T = any>(
  endpoint: string,
  options: ApiRequestOptions = {}
): Promise<ApiResponse<T>> {
  const { method = 'GET', headers = {}, body, ...rest } = options;

  // Prepare headers
  const requestHeaders: HeadersInit = {
    'Content-Type': 'application/json',
    ...headers,
  };

  // Prepare request options
  const requestOptions: RequestInit = {
    method,
    headers: requestHeaders,
    credentials: 'include', // Include cookies in the request
    ...rest,
  };

  // Add body if present
  if (body) {
    requestOptions.body = JSON.stringify(body);
  }

  try {
    // Make the request
    const response = await fetch(`${API_BASE_URL}${endpoint}`, requestOptions);

    // Handle non-JSON responses (like SSE)
    const contentType = response.headers.get('content-type');
    if (contentType?.includes('text/event-stream')) {
      // For SSE, we'll handle it differently in the component
      throw new Error('SSE response should be handled with EventSource');
    }

    // Parse JSON response
    const data = await response.json();

    // Check if the response was successful
    if (!response.ok) {
      throw new Error(data.message || 'API request failed');
    }

    return data;
  } catch (error) {
    console.error('API request error:', error);
    throw error;
  }
}

/**
 * Creates an EventSource for Server-Sent Events
 * @param endpoint - The API endpoint to connect to (without leading slash)
 * @param options - Request options
 * @returns EventSource instance
 */
export function createEventSource(
  endpoint: string,
  options: { method?: string; headers?: HeadersInit; body?: any } = {}
): EventSource {
  const { method = 'GET', headers = {}, body } = options;

  // For SSE, we need to use GET method and pass parameters in the URL
  if (method !== 'GET') {
    console.warn('SSE only supports GET method. Method will be ignored.');
  }

  // Convert body to URL parameters if present
  let url = `${API_BASE_URL}${endpoint}`;
  if (body) {
    const params = new URLSearchParams();
    Object.entries(body).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        params.append(key, String(value));
      }
    });
    url += `?${params.toString()}`;
  }

  // Create EventSource
  return new EventSource(url);
}

/**
 * Helper function to handle SSE events
 * @param eventSource - The EventSource instance
 * @param handlers - Event handlers
 */
export function handleSSEEvents(
  eventSource: EventSource,
  handlers: {
    onMessage?: (data: any) => void;
    onError?: (error: Event) => void;
    onOpen?: (event: Event) => void;
  }
): () => void {
  const { onMessage, onError, onOpen } = handlers;

  // Set up event listeners
  if (onMessage) {
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('Error parsing SSE message:', error);
      }
    };
  }

  if (onError) {
    eventSource.onerror = onError;
  }

  if (onOpen) {
    eventSource.onopen = onOpen;
  }

  // Return cleanup function
  return () => {
    eventSource.close();
  };
} 