import { tokenStorage } from '../lib/tokenStorage';

const API_URL = import.meta.env.VITE_API_URL;

type LoginResponse = {
  access_token: string;
  token_type: string;
};

export async function loginUser(
  login: string,
  password: string
): Promise<LoginResponse> {
  const body = new URLSearchParams();

  body.append('username', login);
  body.append('password', password);

  const response = await fetch(`${API_URL}/users/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body,
  });

  if (!response.ok) {
    throw new Error('Неверный логин или пароль');
  }

  const data: LoginResponse = await response.json();

  tokenStorage.setToken(data.access_token);

  return data;
}

export function logoutUser(): void {
  tokenStorage.removeToken();
}