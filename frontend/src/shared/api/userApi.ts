import { apiClient } from './apiClient';
import type { UserRole } from '../../entities/user/model/types';

export type MyFioRole = {
  fio: string;
  role: UserRole;
};

export type MyShortInfo = {
  fio: string;
  login: string;
  organization_name: string;
};

export type MyFullInfo = MyShortInfo & {
  id: number;
  role: UserRole;
  name: string;
  surname: string;
  patronymic?: string | null;
  date_of_birth: string;
  is_active: boolean;
};

export const getMyFioRole = () => {
  return apiClient<MyFioRole>('/users/me/fio-role');
};

export const getMyShortInfo = () => {
  return apiClient<MyShortInfo>('/users/me/short');
};

export const getMyFullInfo = () => {
  return apiClient<MyFullInfo>('/users/me/full');
};

export const changeMyPassword = (oldPassword: string, newPassword: string) => {
  return apiClient<{ message: string }>('/users/me/change-password', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      old_password: oldPassword,
      new_password: newPassword,
    }),
  });
};
