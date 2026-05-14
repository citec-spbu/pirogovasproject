import { Routes, Route, Navigate } from 'react-router-dom';
import { LoginPage } from '../../pages/LoginPage/LoginPage.tsx';
import { MainPage } from '../../pages/MainPage/MainPage.tsx';

export const AppRouter = () => {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/" element={<MainPage />} />

      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};