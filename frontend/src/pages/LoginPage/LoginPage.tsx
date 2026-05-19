import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from '../../shared/ui/Button/Button';
import { Input } from '../../shared/ui/Input/Input';
import { loginUser } from '../../shared/api/authApi';
import { getUserRoleFromToken } from '../../shared/lib/jwt';
import HomeIcon from '../../shared/assets/icons/homeIcon.svg';
import cls from './LoginPage.module.scss';

export const LoginPage = () => {
  const navigate = useNavigate();

  const [login, setLogin] = useState('');
  const [password, setPassword] = useState('');

  const [loginError, setLoginError] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [serverError, setServerError] = useState('');

  const [isRestoreOpen, setIsRestoreOpen] = useState(false);

  const handleSubmit = async () => {
    let isValid = true;

    if (!login.trim()) {
      setLoginError('Введите логин');
      isValid = false;
    } else {
      setLoginError('');
    }

    if (!password.trim()) {
      setPasswordError('Введите пароль');
      isValid = false;
    } else {
      setPasswordError('');
    }

    if (!isValid) return;

    try {
      setServerError('');

      const data = await loginUser(login, password);
      const role = getUserRoleFromToken(data.access_token);

      if (role === 'admin') {
        navigate('/admin', { replace: true });
      } else {
        navigate('/', { replace: true });
      }
    } catch {
      setServerError('Неверный логин или пароль');
    }
  };

  return (
    <div className={cls.page}>
      <div className={cls.card}>
        <div className={cls.logo}>
          <img className={cls.HomeIcon} src={HomeIcon} alt="HomeIcon" />
        </div>

        <div className={cls.form}>
          <Input
            label="Логин *"
            type="text"
            value={login}
            onChange={setLogin}
            variant="floating"
            error={loginError}
          />

          <Input
            label="Пароль *"
            type="password"
            value={password}
            onChange={setPassword}
            variant="floating"
            error={passwordError}
          />

          {serverError && <div className={cls.error}>{serverError}</div>}

          <Button className={cls.submitButton} type="button" onClick={handleSubmit}>
            Авторизоваться
          </Button>

        </div>
      </div>

    </div>
  );
};