import { Routes, Route, Navigate } from 'react-router-dom';
import { LoginPage } from '../../pages/LoginPage/LoginPage';
import { MainPage } from '../../pages/MainPage/MainPage';
import { AdminPage } from '../../pages/AdminPage/AdminPage';
import { ProtectedRoute } from './ProtectedRoute';

export const AppRouter = () => {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />

      <Route
        path="/"
        element={
          <ProtectedRoute allowedRoles={['user', 'admin']}>
            <MainPage />
          </ProtectedRoute>
        }
      />

      <Route
        path="/admin"
        element={
          <ProtectedRoute allowedRoles={['admin']}>
            <AdminPage />
          </ProtectedRoute>
        }
      />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};