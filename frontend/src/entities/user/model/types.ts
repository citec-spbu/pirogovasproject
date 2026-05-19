export type UserRole = 'admin' | 'user' | 'doctor';

export type UserStatus = 'active' | 'inactive';

export type User = {
  id: string;
  fullName: string;
  lastName?: string;
  firstName?: string;
  middleName?: string;
  birthDate?: string;
  login: string;
  organization: string;
  role: UserRole;
  status: UserStatus;
  password?: string;
};

export type UserFormValues = {
  lastName: string;
  firstName: string;
  middleName: string;
  birthDate: string;
  login: string;
  role: UserRole;
  organization: string;
  password?: string;
};