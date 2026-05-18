import { useState } from 'react';
import { Modal } from '../../shared/ui/Modal/Modal';
import { Button } from '../../shared/ui/Button/Button';
import { Input } from '../../shared/ui/Input/Input';
import CrossIcon from '../../shared/assets/icons/exitIcon.svg';
import cls from './ChangePasswordModal.module.scss';

interface ChangePasswordModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ChangePasswordModal = ({
  isOpen,
  onClose,
}: ChangePasswordModalProps) => {
  const [password, setPassword] = useState('');
  const [passwordError, setPasswordError] = useState('');

  const handleSubmit = () => {
    if (!password.trim()) {
      setPasswordError('Введите новый пароль');
      return;
    }

    setPasswordError('');
    console.log('Новый пароль:', password);

    setPassword('');
    onClose();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className={cls.card}>
        <button type="button" className={cls.closeButton} onClick={onClose}>
          <img src={CrossIcon} alt="" className={cls.closeIcon} />
        </button>

        <h2 className={cls.title}>Придумайте новый пароль</h2>

        <div className={cls.formRow}>
          <p className={cls.label}>Новый пароль</p>

          <div className={cls.inputWrapper}>
            <Input
              label=""
              type="password"
              value={password}
              onChange={setPassword}
              variant="default"
              placeholder="user_password"
              error={passwordError}
            />
          </div>
        </div>

        <Button
          type="button"
          className={cls.submitButton}
          onClick={handleSubmit}
        >
          Сохранить
        </Button>
      </div>
    </Modal>
  );
};