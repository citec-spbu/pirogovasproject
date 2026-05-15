export type JwtPayload = {
    sub: string;
    role?: string;
    organization_name?: string;
    exp?: number;
  };
  
  export function decodeJwtPayload(token: string): JwtPayload | null {
    try {
      const payload = token.split('.')[1];
  
      if (!payload) {
        return null;
      }
  
      const normalizedPayload = payload.replace(/-/g, '+').replace(/_/g, '/');
      const decodedPayload = atob(normalizedPayload);
  
      return JSON.parse(decodedPayload) as JwtPayload;
    } catch {
      return null;
    }
  }
  
  export function getUserRoleFromToken(token: string): string | null {
    const payload = decodeJwtPayload(token);
  
    return payload?.role ? payload.role.toLowerCase() : null;
  }
  
  export function isTokenExpired(token: string): boolean {
    const payload = decodeJwtPayload(token);
  
    if (!payload?.exp) {
      return false;
    }
  
    const currentTimeInSeconds = Math.floor(Date.now() / 1000);
  
    return payload.exp < currentTimeInSeconds;
  }