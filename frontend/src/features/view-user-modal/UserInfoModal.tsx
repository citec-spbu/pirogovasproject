import { Modal } from '../../shared/ui/Modal/Modal';
import type { User } from '../../entities/user/model/types';
import userLogo from '../../shared/assets/icons/userLogoW.svg';
import crossIcon from '../../shared/assets/icons/crossIcon.svg';
import cls from './UserInfoModal.module.scss';

interface UserInfoModalProps {
  isOpen: boolean;
  user: User | null;
  onClose: () => void;
  onEdit?: (user: User) => void;
}

const roleLabels: Record<User['role'], string> = {
  admin: 'Администратор',
  user: 'Пользователь',
  doctor: 'Врач',
};

const statusLabels: Record<User['status'], string> = {
  active: 'active',
  inactive: 'inactive',
};

export const UserInfoModal = ({
  isOpen,
  user,
  onClose,
  
}: UserInfoModalProps) => {
  if (!user) {
    return null;
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose}>
      <div className={cls.card}>
        <div className={cls.containerTitl}>
          <div className={cls.containerTitleLogo}>
            <img src={userLogo} alt="" className={cls.userLogo} />
            <h2 className={cls.title}>Информация о пользователе</h2>
          </div>

          <button type="button" className={cls.closeButton} onClick={onClose}>
            <img src={crossIcon} alt="" className={cls.crossIcon} />
          </button>
        </div>

        <div className={cls.content}>
          <div className={cls.mainInfo}>
            <div>
              <p className={cls.fullName}>{user.fullName}</p>
              <p className={cls.role}>{roleLabels[user.role]}</p>
            </div>

            <p className={cls.status}>
              Статус ({statusLabels[user.status]})
            </p>
          </div>

          <div className={cls.infoBox}>
            <div className={cls.row}>
              <span>Логин</span>
              <strong>{user.login}</strong>
            </div>

            <div className={cls.row}>
              <span>Организация</span>
              <strong>{user.organization}</strong>
            </div>

            {user.birthDate && (
              <div className={cls.row}>
                <span>Дата рождения</span>
                <strong>{user.birthDate}</strong>
              </div>
            )}

            <div className={cls.row}>
              <span>ID</span>
              <strong>{user.id}</strong>
            </div>
          </div>

          
        </div>
      </div>
    </Modal>
  );
};