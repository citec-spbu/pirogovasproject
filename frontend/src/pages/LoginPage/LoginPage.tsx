import { useState } from 'react';
import { Button } from '../../shared/ui/Button/Button';
import { Input } from '../../shared/ui/Input/Input';
import { RestorePasswordModal } from '../../features/restore-password/RestorePasswordModal';
import HomeIcon from '../../shared/assets/icons/homeIcon.svg';
import InfoIcon from '../../shared/assets/icons/infoIcon.svg';
import cls from './LoginPage.module.scss';

export const LoginPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const [emailError, setEmailError] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const [isRestoreOpen, setIsRestoreOpen] = useState(false);

  const handleSubmit = () => {
    let isValid = true;

    if (!email.trim()) {
      setEmailError('Введите email');
      isValid = false;
    } else {
      setEmailError('');
    }

    if (!password.trim()) {
      setPasswordError('Введите пароль');
      isValid = false;
    } else {
      setPasswordError('');
    }

    if (!isValid) return;

    console.log('Форма отправлена');
  };

  return (
    <div className={cls.page}>
      <div className={cls.card}>
        <div className={cls.logo}>
          <img className={cls.HomeIcon} src={HomeIcon} alt="HomeIcon" />
        </div>

        <div className={cls.form}>
          <Input
            label="Login *"
            type="email"
            value={email}
            onChange={setEmail}
            variant="floating"
            error={emailError}
          />

          <Input
            label="Пароль *"
            type="password"
            value={password}
            onChange={setPassword}
            variant="floating"
            error={passwordError}
          />

          <Button className={cls.submitButton} type="button" onClick={handleSubmit}>
            Авторизоваться
          </Button>

          <button
            type="button"
            className={cls.forgotButton}
            onClick={() => setIsRestoreOpen(true)}
          >
            Забыли пароль?
          </button>
        </div>

        <div className={cls.hint}>
          <span className={cls.infoIcon}>
            <img className={cls.InfoIcon} src={InfoIcon} alt="Info" />
          </span>
          <span>
            Если вы впервые, нажмите “Забыли пароль?”, чтобы установить пароль
          </span>
        </div>
      </div>

      <RestorePasswordModal
        isOpen={isRestoreOpen}
        onClose={() => setIsRestoreOpen(false)}
      />
    </div>
  );
};
