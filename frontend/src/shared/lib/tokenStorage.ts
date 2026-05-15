const ACCESS_TOKEN_KEY = 'access_token';

export const tokenStorage = {
    
  getToken(): string | null {
    return localStorage.getItem(ACCESS_TOKEN_KEY);
  },

  setToken(token: string): void {
    localStorage.setItem(ACCESS_TOKEN_KEY, token);
  },

  removeToken(): void {
    localStorage.removeItem(ACCESS_TOKEN_KEY);
  },

  isAuthorized(): boolean {
    return Boolean(localStorage.getItem(ACCESS_TOKEN_KEY));
  },

};