import { useEffect, useState } from 'react';
import { Modal } from '../../shared/ui/Modal/Modal';
import type {
  User,
  UserFormValues,
  UserRole,
} from '../../entities/user/model/types';
import { Button } from '../../shared/ui/Button/Button';
import userLogo from '../../shared/assets/icons/userLogoW.svg';
import crossIcon from '../../shared/assets/icons/crossIcon.svg';
import cls from './UserFormModal.module.scss';

interface UserFormModalProps {
  isOpen: boolean;
  title: string;
  submitText: string;
  initialUser?: User | null;
  passwordLabel?: string;
  isPasswordRequired?: boolean;
  onClose: () => void;
  onSubmit: (values: UserFormValues) => void;
}

const emptyValues: UserFormValues = {
  lastName: '',
  firstName: '',
  middleName: '',
  birthDate: '',
  login: '',
  role: 'user',
  organization: '',
  password: '',
};

export const UserFormModal = ({
  isOpen,
  title,
  submitText,
  initialUser,
  passwordLabel = 'Пароль',
  isPasswordRequired = false,
  onClose,
  onSubmit,
}: UserFormModalProps) => {
  const [values, setValues] = useState<UserFormValues>(emptyValues);
  const [error, setError] = useState('');

  useEffect(() => {
    if (!isOpen) return;

    if (initialUser) {
      setValues({
        lastName: initialUser.lastName ?? '',
        firstName: initialUser.firstName ?? '',
        middleName: initialUser.middleName ?? '',
        birthDate: initialUser.birthDate ?? '',
        login: initialUser.login,
        role: initialUser.role,
        organization: initialUser.organization,
        password: '',
      });
    } else {
      setValues(emptyValues);
    }

    setError('');
  }, [isOpen, initialUser]);

  const updateField = <K extends keyof UserFormValues>(
    field: K,
    value: UserFormValues[K]
  ) => {
    setValues((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSubmit = () => {
    if (
      !values.lastName.trim() ||
      !values.firstName.trim() ||
      !values.login.trim() ||
      !values.organization.trim()
    ) {
      setError('Заполните обязательные поля');
      return;
    }

    if (isPasswordRequired && !values.password?.trim()) {
      setError('Введите пароль');
      return;
    }

    setError('');
    onSubmit(values);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className={cls.card}>
        <div className={cls.containerTitl}>
          <div className={cls.containerTitleLogo}>
            <img src={userLogo} alt="" className={cls.userLogo} />
            <h2 className={cls.title}>{title}</h2>
          </div>

          <button type="button" className={cls.closeButton} onClick={onClose}>
            <img src={crossIcon} alt="" className={cls.crossIcon} />
          </button>
        </div>

        <div className={cls.body}>
          <section className={cls.section}>
            <h3 className={cls.sectionTitle}>Личные данные</h3>

            <div className={cls.box}>
              <label className={cls.row}>
                <span>Фамилия</span>
                <input
                  value={values.lastName}
                  onChange={(event) =>
                    updateField('lastName', event.target.value)
                  }
                  placeholder="Иванов"
                />
              </label>

              <label className={cls.row}>
                <span>Имя</span>
                <input
                  value={values.firstName}
                  onChange={(event) =>
                    updateField('firstName', event.target.value)
                  }
                  placeholder="Иван"
                />
              </label>

              <label className={cls.row}>
                <span>Отчество</span>
                <input
                  value={values.middleName}
                  onChange={(event) =>
                    updateField('middleName', event.target.value)
                  }
                  placeholder="Иванович"
                />
              </label>

              <label className={cls.row}>
                <span>Дата рождения</span>
                <input
                  value={values.birthDate}
                  onChange={(event) =>
                    updateField('birthDate', event.target.value)
                  }
                  placeholder="XX.XX.XXXX"
                />
              </label>
            </div>
          </section>

          <section className={cls.section}>
            <h3 className={cls.sectionTitle}>Аккаунт и доступ</h3>

            <div className={cls.box}>
              <label className={cls.row}>
                <span>Логин</span>
                <input
                  value={values.login}
                  onChange={(event) => updateField('login', event.target.value)}
                  placeholder="user_login"
                />
              </label>

              <label className={cls.row}>
                <span>Роль</span>
                <select
                  value={values.role}
                  onChange={(event) =>
                    updateField('role', event.target.value as UserRole)
                  }
                >
                  <option value="user">Пользователь</option>
                  <option value="admin">Администратор</option>
                </select>
              </label>

              <label className={cls.row}>
                <span>Организация</span>
                <input
                  value={values.organization}
                  onChange={(event) =>
                    updateField('organization', event.target.value)
                  }
                  placeholder="user_org"
                />
              </label>
            </div>
          </section>

          <section className={cls.section}>
            <h3 className={cls.sectionTitle}>Пароль</h3>

            <div className={cls.box}>
              <label className={cls.row}>
                <span>{passwordLabel}</span>
                <input
                  type="password"
                  value={values.password ?? ''}
                  onChange={(event) =>
                    updateField('password', event.target.value)
                  }
                  placeholder="user_password"
                />
              </label>
            </div>
          </section>

          {error && <p className={cls.error}>{error}</p>}

          <Button type="button" className={cls.submitButton} onClick={handleSubmit} >
            {submitText}
          </Button>
        </div>
      </div>
    </Modal>
  );
};