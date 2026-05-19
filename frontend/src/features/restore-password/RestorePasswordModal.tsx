import { useState } from 'react';
import { Modal } from '../../shared/ui/Modal/Modal';
import { Button } from '../../shared/ui/Button/Button';
import { Input } from '../../shared/ui/Input/Input';
import cls from './RestorePasswordModal.module.scss';
import ExitIcon from '../../shared/assets/icons/exitIcon.svg';

interface RestorePasswordModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const RestorePasswordModal = ({
  isOpen,
  onClose,
}: RestorePasswordModalProps) => {
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');

  const handleSubmit = () => {
    if (!email.trim()) {
      setEmailError('Введите email');
      return;
    }

    setEmailError('');
    console.log('Восстановление пароля');
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className={cls.card}>
        <button type="button" className={cls.closeButton} onClick={onClose}>
            <img className={cls.ExitIcon} src={ExitIcon} alt="HomeIcon" />
        </button>

        <h2 className={cls.title}>ВОССТАНОВЛЕНИЕ ПАРОЛЯ</h2>

        <div className={cls.form}>
          <Input
            label="Введите email:"
            type="email"
            value={email}
            onChange={setEmail}
            variant="default"
            placeholder="email@example.com"
            error={emailError}
          />

          <Button type="button" className={cls.submitButton} onClick={handleSubmit}>
            Восстановить
          </Button>
        </div>
      </div>
    </Modal>
  );
};