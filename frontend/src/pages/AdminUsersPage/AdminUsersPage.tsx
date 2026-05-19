import { useMemo, useState } from 'react';
import type { User, UserFormValues } from '../../entities/user/model/types';
import { AdminUsersToolbar } from '../../widgets/AdminUsersToolbar/AdminUsersToolbar';
import { UsersTable } from '../../widgets/UsersTable/UsersTable';
import { UserFormModal } from '../../features/user-form-modal/UserFormModal';
import { UserInfoModal } from '../../features/view-user-modal/UserInfoModal';
import addIcon from '../../shared/assets/icons/addIcon.svg';
import cls from './AdminUsersPage.module.scss';

const mockUsers: User[] = [
  {
    id: '1',
    fullName: 'Вячеславов Вячеслав Вячеславович',
    lastName: 'Вячеславов',
    firstName: 'Вячеслав',
    middleName: 'Вячеславович',
    birthDate: '01.01.1990',
    login: 'user_login',
    organization: 'Название',
    role: 'user',
    status: 'active',
  },
  {
    id: '2',
    fullName: 'Иванов Иван Иванович',
    lastName: 'Иванов',
    firstName: 'Иван',
    middleName: 'Иванович',
    birthDate: '02.02.1991',
    login: 'ivanov',
    organization: 'Поликлиника №1',
    role: 'doctor',
    status: 'active',
  },
  {
    id: '3',
    fullName: 'Сергей Ли',
    lastName: 'Ли',
    firstName: 'Сергей',
    middleName: '',
    birthDate: '03.03.1992',
    login: 'koreya',
    organization: 'Неколаевская',
    role: 'doctor',
    status: 'active',
  },
];

const buildFullName = (values: UserFormValues): string => {
  return [values.lastName, values.firstName, values.middleName]
    .filter(Boolean)
    .join(' ');
};

export const AdminUsersPage = () => {
  const [users, setUsers] = useState<User[]>(mockUsers);
  const [searchValue, setSearchValue] = useState('');

  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [viewUser, setViewUser] = useState<User | null>(null);
  const [editUser, setEditUser] = useState<User | null>(null);

  const filteredUsers = useMemo(() => {
    const normalizedSearch = searchValue.trim().toLowerCase();

    if (!normalizedSearch) {
      return users;
    }

    return users.filter((user) => {
      return (
        user.fullName.toLowerCase().includes(normalizedSearch) ||
        user.login.toLowerCase().includes(normalizedSearch) ||
        user.organization.toLowerCase().includes(normalizedSearch)
      );
    });
  }, [searchValue, users]);

  const handleSearchSubmit = () => {
    console.log('Поиск пользователя:', searchValue);
  };

  const handleCreateUser = (values: UserFormValues) => {
    const newUser: User = {
      id: crypto.randomUUID(),
      fullName: buildFullName(values),
      lastName: values.lastName,
      firstName: values.firstName,
      middleName: values.middleName,
      birthDate: values.birthDate,
      login: values.login,
      organization: values.organization,
      role: values.role,
      status: 'active',
    };

    setUsers((prevUsers) => [...prevUsers, newUser]);
    setIsCreateOpen(false);
  };

  const handleEditUser = (values: UserFormValues) => {
    if (!editUser) {
      return;
    }

    setUsers((prevUsers) =>
      prevUsers.map((user) =>
        user.id === editUser.id
          ? {
              ...user,
              fullName: buildFullName(values),
              lastName: values.lastName,
              firstName: values.firstName,
              middleName: values.middleName,
              birthDate: values.birthDate,
              login: values.login,
              organization: values.organization,
              role: values.role,
            }
          : user
      )
    );

    setEditUser(null);
  };

  const handleDeleteUser = (user: User) => {
    console.log('Удалить пользователя:', user);
  };

  return (
    <div className={cls.page}>
      <AdminUsersToolbar
        searchValue={searchValue}
        onSearchChange={setSearchValue}
        onSearchSubmit={handleSearchSubmit}
      />

      <section className={cls.section}>
        <div className={cls.sectionHeader}>
          <h1 className={cls.title}>Все пользователи</h1>

          <button
            type="button"
            className={cls.createButton}
            onClick={() => setIsCreateOpen(true)}
            aria-label="Создать пользователя"
          >
            <img
              src={addIcon}
              alt=""
              className={cls.addIcon}
            />
          </button>
        </div>

        <UsersTable
          users={filteredUsers}
          onViewUser={setViewUser}
          onEditUser={setEditUser}
          onDeleteUser={handleDeleteUser}
        />
      </section>

      <UserFormModal
        isOpen={isCreateOpen}
        title="Создание пользователя"
        submitText="Создать"
        passwordLabel="Пароль"
        isPasswordRequired
        onClose={() => setIsCreateOpen(false)}
        onSubmit={handleCreateUser}
      />

      <UserFormModal
        isOpen={Boolean(editUser)}
        title="Редактирование пользователя"
        submitText="Сохранить"
        initialUser={editUser}
        passwordLabel="Новый пароль"
        isPasswordRequired={false}
        onClose={() => setEditUser(null)}
        onSubmit={handleEditUser}
      />

      <UserInfoModal
        isOpen={Boolean(viewUser)}
        user={viewUser}
        onClose={() => setViewUser(null)}
      />
    </div>
  );
};