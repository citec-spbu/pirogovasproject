import { tokenStorage } from '../lib/tokenStorage';

const API_URL = import.meta.env.VITE_API_URL;

type ApiClientOptions = RequestInit & {
  auth?: boolean;
};

async function getErrorMessage(response: Response): Promise<string> {
  try {
    const data = await response.json();

    if (typeof data.detail === 'string') {
      return data.detail;
    }

    return `Ошибка запроса: ${response.status}`;
  } catch {
    return `Ошибка запроса: ${response.status}`;
  }
}

export async function apiClient<T>(
  endpoint: string,
  options: ApiClientOptions = {}
): Promise<T> {
  const { auth = true, headers, ...restOptions } = options;

  const token = tokenStorage.getToken();

  const response = await fetch(`${API_URL}${endpoint}`, {
    ...restOptions,
    headers: {
      ...(auth && token ? { Authorization: `Bearer ${token}` } : {}),
      ...headers,
    },
  });

  if (response.status === 401) {
    tokenStorage.removeToken();
    throw new Error('Необходима повторная авторизация');
  }

  if (response.status === 403) {
    throw new Error('Недостаточно прав доступа');
  }

  if (!response.ok) {
    const message = await getErrorMessage(response);
    throw new Error(message);
  }

  return response.json() as Promise<T>;
}