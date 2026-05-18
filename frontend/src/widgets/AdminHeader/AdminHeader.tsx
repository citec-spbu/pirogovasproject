import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ProfileDropdown } from '../ProfileDropdown/ProfileDropdown';
import { ChangePasswordModal } from '../../features/change-password/ChangePasswordModal';
import { logoutUser } from '../../shared/api/authApi';
import HeartlogoIcon from '../../shared/assets/icons/heartLogoIcon.svg';
import DoorIcon from '../../shared/assets/icons/doorIcon.svg';
import cls from './AdminHeader.module.scss';

export const AdminHeader = () => {
  const navigate = useNavigate();
  const [isChangePasswordOpen, setIsChangePasswordOpen] = useState(false);

  const handleLogout = () => {
    logoutUser();
    navigate('/login', { replace: true });
  };

  const handleChangePassword = () => {
    setIsChangePasswordOpen(true);
  };

  const handleGoToUserPage = () => {
    navigate('/');
  };

  return (
    <>
      <header className={cls.header}>
        <div className={cls.logoBlock}>
          <img src={HeartlogoIcon} alt="CTReport Admin" className={cls.logoIcon} />
          <span className={cls.logoText}>CTReport Admin</span>
        </div>

        <div className={cls.actions}>
          <ProfileDropdown
            triggerLabel="Пользователь"
            fullName="Фамилия Имя Отчество"
            roleLabel="Администратор"
            showChangePassword
            onChangePassword={handleChangePassword}
            onLogout={handleLogout}
          />

          <button
            type="button"
            className={cls.userPageButton}
            onClick={handleGoToUserPage}
            aria-label="Перейти в пользовательскую часть"
          >
            <img src={DoorIcon} alt="" className={cls.doorIcon} />
          </button>
        </div>
      </header>

      <ChangePasswordModal
        isOpen={isChangePasswordOpen}
        onClose={() => setIsChangePasswordOpen(false)}
      />
    </>
  );
};