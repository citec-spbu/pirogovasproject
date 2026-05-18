import { Navigate, Route, Routes } from 'react-router-dom';
import { LoginPage } from '../../pages/LoginPage/LoginPage';
import { MainPage } from '../../pages/MainPage/MainPage';
import { AdminDashboardPage } from '../../pages/AdminDashboardPage/AdminDashboardPage';
import { AdminUsersPage } from '../../pages/AdminUsersPage/AdminUsersPage.tsx';
import { AdminTemplatesPage } from '../../pages/AdminTemplatesPage/AdminTemplatesPage';
import { AdminProtocolsPage } from '../../pages/AdminProtocolsPage/AdminProtocolsPage';
import { ProtectedRoute } from './ProtectedRoute';
import { AdminLayout } from '../layouts/AdminLayout/AdminLayout';


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
            <AdminLayout />
          </ProtectedRoute>
        }
      >
        <Route index element={<AdminDashboardPage />} />
        <Route path="users" element={<AdminUsersPage />} />
        <Route path="templates" element={<AdminTemplatesPage />} />
        <Route path="protocols" element={<AdminProtocolsPage />} />
      </Route>

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};