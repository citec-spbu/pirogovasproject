import type { ReactElement } from 'react';
import { Navigate } from 'react-router-dom';
import { tokenStorage } from '../../shared/lib/tokenStorage';
import { getUserRoleFromToken, isTokenExpired } from '../../shared/lib/jwt';

const AUTH_BYPASS = import.meta.env.VITE_AUTH_BYPASS === 'true';

type ProtectedRouteProps = {
  children: ReactElement;
  allowedRoles?: string[];
};

export const ProtectedRoute = ({
  children,
  allowedRoles,
}: ProtectedRouteProps): ReactElement => {
  console.log('AUTH_BYPASS:', AUTH_BYPASS);

  if (AUTH_BYPASS) {
    return children;
  }

  const token = tokenStorage.getToken();

  if (!token || isTokenExpired(token)) {
    tokenStorage.removeToken();
    return <Navigate to="/login" replace />;
  }

  if (allowedRoles && allowedRoles.length > 0) {
    const role = getUserRoleFromToken(token);

    if (!role || !allowedRoles.includes(role)) {
      return <Navigate to="/" replace />;
    }
  }

  return children;
};